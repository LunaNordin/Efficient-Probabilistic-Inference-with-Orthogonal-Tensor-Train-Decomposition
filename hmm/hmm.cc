#include "hmm.h"

using namespace itensor;
using namespace std;

/**
 * Generates a hidden markov model with specified dimensions and random values
 * hiddenDim: dimension of the one hidden variable
 * visibleVariables: number of visible variables
 * visibleDim: dimension of all visible variables
 * mode: which representation of the emission tensor to use (saves memory space if certain representations are not needed)
 * return: HMM object with specified number of varibales and dimensions as well as random values in transition, emision and initial state
*/
HMM generate_hmm(int hiddenDim, int visibleVariables, int visibleDim, model_mode mode) {
    // model to be filled with data
    HMM model;
    
    // set the number of variables and their dimensions
    model.hiddenVariables = 1;  // there is always only one hidden variable
    model.hiddenDimension = hiddenDim;
    model.visibleVariables = visibleVariables;
    model.visibleDimension = visibleDim;

    // create a matrix (rank-2) tensor with random values as transition matrix
    auto j = Index(hiddenDim);
    auto k = Index(hiddenDim);
    model.transition = randomITensor(j,k);

    // TODO: currently vsibleDim and hiddenDim have to be indentical because emission tensor has to be symmetric

    // generate a symmetric, odeco tensor train as the mps representation of the emission tensor 
    // has rank of one more than number of visible variables because states of hidden variable are encoded as well
    auto emission_mps = generate_symmetric_odeco_tensor_train(visibleVariables + 1, visibleDim);
    // set the mps representation if it is required
    if(mode == mps || mode == both_models) {
        model.emission_mps = emission_mps;
    }

    // set the tensor representation if it is required
    if(mode == tensor || mode == both_models) {
        // contract that tensor train and also save the contracted emission tensor
        model.emission_tensor = contract_tensor_train(emission_mps);
    }

    // add the initial state to the model
    model.initial_state = generate_state(visibleVariables, visibleDim); 
    // add the initial probabilities of the hidden variable
    auto l = Index(hiddenDim);
    model.initial_hidden_probability = randomITensor(l);

    // return the generated model
    return model;
}

/**
 * Generates a sequence of random states which can be used as evidence
 * visibleVariables: Number of values in each state
 * visibleDim: value range of each state component
 * length: number of generated states
 * return: array of states in form of vektors
*/
vector<int>* generate_state_sequence(int visibleVariables, int visibleDim, int length) {
    // set up an array of vectors to hold the state sequence
    vector<int>* sequence = new vector<int>[length];

    // generate lenght many states
    for (int i = 1; i <= length; i++) {
        // generate a random state
        vector<int> state = generate_state(visibleVariables, visibleDim);
        // add the state to the sequence
        sequence[i-1] = state;
    }
    // return the generated state sequence
    return sequence;
}

/**
 * Generates a random state for a model with specified parameters
 * visibleVariables: number of variables for which there are states to be generated
 * visibleDim: dimension of the variables is range for number generation
 * return: vector of specified dimensions containing generated state
*/
vector<int> generate_state(int visibleVariables, int visibleDim) {
    // set up a random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(1,visibleDim);

    // create a vector for the state
    vector<int> state;

    // fill the state vector with random values from the dimension value range
    for (int j = 1; j <= visibleVariables; j++) {
        auto random = uni(rng);
        state.push_back(random);
    }
    // return the generated state
    return state;
}

/**
 * Calculates estimated memory usage for the representation of an HMM and reports if it can be stored in system memory
 * total_variables: total number of variables (hidden and visible)
 * dimension: dimension of the variables
 * mode: which representation of the emission tensor is used
 * return: 1 if representation exceeds memory limit, 1 if its greater than 75%, 0 else
*/
int has_critical_memory_demand(int total_variables, int dimension, model_mode mode) {
    float system_memory = 12.0;

    // number of bytes of the tensor
    float tensor_size_bytes = (pow(dimension, total_variables)) * sizeof(long);
    // size in GB of the tensor
    float tensor_size_gigabytes = tensor_size_bytes / float(pow(10, 9));

    // number of bytes of the mps
    float train_size_bytes = (total_variables*dimension*pow(dimension, 2)) * sizeof(long);
    // size in GB of the mps
    float train_size_gigabytes = train_size_bytes / float(pow(10, 9));

    // total size of mps
    float total_size = 0;
    if(mode == mps) {
        total_size = train_size_gigabytes;
    } else if (mode == tensor) {
        total_size = tensor_size_gigabytes;
    } else if (mode == both_models) {
        total_size = train_size_gigabytes + tensor_size_gigabytes;
    }
        
    // return if system memory is sufficient
    if(total_size >= system_memory) {
        // size is greater than system memory
        return 2;
    } else if (total_size >= system_memory * 0.75){
        // size is smaller than system memory but at least 75% of system memory
        return 1;
    } else {
        // size is smaller than system memory
        return 0;
    }
}

/**
 * Calculates a component in a tensor from its MPS representation for a given set of indices
 * model: HMM containing the tensor and the MPS representation 
 * evidence: list of indices (evidence in case of an HMM, visible state at first place) specifying a component in the tensor
 * parallel: options for parallelization
 * return: component corresponding to the given indices
*/
Real get_component_from_tensor_train(HMM model, vector<int> evidence, ParallelizationOpt parallel) {
    // TODO: make sure length of evidence sequence matches number of visible variables in train
    
    ITensor component;  // tensor which will contain the calculated emission value
    
    // absorb the evidence by removing all non-fitting fields from the mps
    if(parallel.mode == sequential || parallel.mode == parallel_evidence) {
        // no parallelization of evidence absorption
        absorb_evidence(model, evidence, length(model.emission_mps), 1);

        // contract the train to calculate the field from the tensor corresponding to the absorbed evidence
        component = contract_tensor_train(model.emission_mps);
    } else if(parallel.mode == parallel_contraction || parallel.mode == both_parallel) {
        // split absorption of evidence into two threads
        
        // split length of mps into two halfs
        int length_1 = length(model.emission_mps) / 2;
        int length_2 = length(model.emission_mps) - length_1;
        // calculate start points for each of the threads
        int start_1 = 1;
        int start_2 = length_1 + 1;
        // start two threads, each of which will absorb the evidence in half of the carriages
        thread th1(absorb_evidence, std::ref(model), evidence, length_1, start_1);
        thread th2(absorb_evidence, std::ref(model), evidence, length_2, start_2);
        // wait until both threads have finished running
        th1.join();
        th2.join();
        // start two threads that each multiply the carriage in one half of the train
        auto contract_left_future = std::async(contract_tensor_train_parallel, model.emission_mps, length_1, start_1);
        auto contract_right_future = std::async(contract_tensor_train_parallel, model.emission_mps, length_2, start_2);
        // wait for both threads to return their results
        auto contract_left = contract_left_future.get();
        auto contract_right = contract_right_future.get();
        // multiply the two contracted halfs to calculate the emission value
        component = contract_left * contract_right;
    }
    
    // return the single value from the rank-0 tensor
    return component.elt();
}

/**
 * Calculates a component in a tensor from its MPS representation for a given set of indices and makes sure the result matches the expected value
 * model: HMM containing the tensor and the MPS representation 
 * evidence: list of indices (evidence in case of an HMM, visible state at first place) specifying a component in the tensor
 * parallel: options for parallelization
 * return: component corresponding to the given indices or -1 if deviation from expected result is too high
*/
Real get_component_from_tensor_train_with_check(HMM model, vector<int> evidence, ParallelizationOpt parallel) {
    // maximum deviation from expected result allowed (some accuracy might be lost during calculation)
    double max_error = 0.00001;
    // calculate component from MPS
    auto component = get_component_from_tensor_train(model, evidence, parallel);
    // get expected component from tensor
    auto control = elt(model.emission_tensor, evidence);
    // make sure the two are identical (within expected accuracy)
    if(abs(component - control) > max_error) {
        // infrom user about deviation and return error value
        println("Error: Component calculated from MPS representation does not match tensor component");
        return -1;
    }

    // return the calculated component
    return component;
}

/**
 * Absorbs the given evidence on the specified section of the tensor train
 * model: HMM containing the tensor and the MPS representation 
 * evidence: list of indices (evidence in case of an HMM, visible state at first place) specifying a component in the tensor
 * length: number of carriages to absorb the evidence into
 * start: index of carriage to start at
*/
void absorb_evidence(HMM& model, vector<int> evidence, int length, int start) {
    // absorb the evidence in every carraige of the train
    for(int i = 0; i < length; i++) {

        ITensor prev_carriage;          // carriage at position before the current carriage in the train
        ITensor curr_carriage;          // carriage the current position in the train
        ITensor next_carriage;          // carriage at position after the current carriage in the train
        vector<ITensor> neighbours;     // vector containing existing neighbours (one or two)
        IndexSet bond_left;             // set containing the left bond_index (if it exists)
        IndexSet bond_right;            // set containing the right bond_index (if it exists)

        // get the current carriage
        curr_carriage = model.getCarriageFromTrain(start + i);
        // if a left neighbour exists (not at the left end of the train)
        if(start + i != 1) {
            // get the left neighbour carriage
            prev_carriage = model.getCarriageFromTrain(start + i - 1);
            // add the carriage to the set of existing neighbours
            neighbours.push_back(prev_carriage);
            // get the index shared between the current carriage and its left neighbour
            bond_left = commonInds(curr_carriage, prev_carriage);
        }
        // if a right neighbour exists (not at the right end of the train)
        if(start + i != model.emission_mps.length()) {
            // get the right neighbour carriage
            next_carriage = model.getCarriageFromTrain(start + i + 1);
            // add the carriage to the set of existing neighbours
            neighbours.push_back(next_carriage);
            // get the index shared between the current carriage and its right neighbour
            bond_right = commonInds(curr_carriage, next_carriage);
        }

        // the visible indices are the only ones not shared with the existing neighbours (at least one, up to two)
        auto visible_indices = uniqueInds(curr_carriage, neighbours);
        // the hidden indices are the ones shared with the existing neighbours (at least one, up to two)
        auto hidden_indices = IndexSet(bond_left, bond_right);

        // create a new carriage only containing the bond indices
        auto new_carriage = ITensor(hidden_indices);

        // copy over all values which match the evidence to the new carriage
        auto bond_1 = hidden_indices.front();   // there is always at least one bond index
        auto visible_1 = visible_indices.front();   // there is always at least one visible index
        // cycle over the one bond index which is sure to exist
        for(int j = 1; j <= dim(bond_1); j++) {
            // there might be a second bond index if the current carriage is not an end of the train
            if(start + i != 1 && start + i != model.emission_mps.length()) {
                auto bond_2 = hidden_indices.back();    // in the middle there are two bond indices
                // cycle over the second bond index
                for(int k = 1; k <= dim(bond_1); k++) {
                    // get the field where the one visible index matches the given evidence
                    auto field = curr_carriage.elt(visible_1 = evidence.at(start + i), bond_1 = j, bond_2 = k);
                    // copy it over to the new tensor
                    new_carriage.set(bond_1 = j, bond_2 = k, field);
                }
            } else if(start + i == 1) {
                // first carriage in the train
                auto visible_2 = visible_indices.back(); // at the ends there are two visible indices
                // get the field where the two visible indices match the given evidence
                auto field = curr_carriage.elt(visible_1 = evidence.at(start + i - 1), visible_2 = evidence.at(start + i), bond_1 = j);
                // copy it over to the new tensor
                new_carriage.set(bond_1 = j, field);
            } else {
                // last carriage in the train
                auto visible_2 = visible_indices.back(); // at the ends there are two visible indices
                // get the field where the two visible indices match the given evidence
                auto field = curr_carriage.elt(visible_1 = evidence.at(start + i), visible_2 = evidence.at(start + i + 1), bond_1 = j);
                // copy it over to the new tensor
                new_carriage.set(bond_1 = j, field);
            }
        }
        // replace the old carrige with the new carriage which has the evidence absorbed
        model.setTrainCarriage(start + i, new_carriage);
    }
}

/**
 * Gives a string represenation for any model_mode value
 * mode: mode to be converted to string
 * return: name of the mode as a string
*/
string mode_to_string(model_mode mode)
{
    // TODO: replace this, its a maintenance nightmare
    switch (mode)
    {
        case tensor:            return "tensor";
        case mps:               return "mps";
        case both_models:       return "both_models";
        default:                return "unknown model_mode";
    }
}

/**
 * Used to test the generation of hmms
*/
int test_hmm() {
    // test the model generation
    HMM model = generate_hmm(3, 4, 3, both_models);
    println("Generated HMM with hidden dimension 3 and four hidden variables of dimension 3:");
    println("-------------------------------------------------------------------------------");

    println("Transition matrix:");
    PrintData(model.transition);

    println("Emission tensor:");
    PrintData(model.emission_tensor);

    println("Emission tensor train:");
    PrintData(model.emission_mps);

    println("Initial state:");
    auto state = model.initial_state;
    copy(state.begin(), state.end(), ostream_iterator<int>(cout, " "));
    cout << "\n\n"; 

    println("Initial hidden probabilities:");
    PrintData(model.initial_hidden_probability);

    println("Random state sequence:");
    auto sequence = generate_state_sequence(model.visibleVariables, model.visibleDimension, 4);
    for (int i = 0; i < 4; i++) {
        copy(sequence[i].begin(), sequence[i].end(), ostream_iterator<int>(cout, " "));
        cout << "  "; 
    }
    cout << "\n\n"; 

    println("Get random component from train:");
    auto rand_state = generate_state(model.visibleVariables + 1, model.visibleDimension);
    auto component = get_component_from_tensor_train(model, rand_state, ParallelizationOpt(sequential));
    print("Calculated from MPS: ");
    println(component);
    print("Taken from tensor: ");
    println(elt(model.emission_tensor, rand_state));
    get_component_from_tensor_train_with_check(model, rand_state, ParallelizationOpt(sequential));

    return 0;
}