#include "hmm.h"

using namespace itensor;
using namespace std;

/**
 * Generates a hidden markov model with specified dimensions and random values
 * hiddenDim: dimension of the one hidden variable
 * visibleVariables: number of visible variables
 * visibleDim: dimension of all visible variables
 * return: HMM object with specified number of varibales and dimensions as well as random values in transition, emision and initial state
*/
HMM generate_hmm(int hiddenDim, int visibleVariables, int visibleDim) {
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
    model.emission_mps = generate_symmetric_odeco_tensor_train(visibleVariables + 1, visibleDim);
    // contract that tensor train and also save the contracted emission tensor
    model.emission_tensor = contract_tensor_train(model.emission_mps);

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
 * return: 1 if representation exceeds memory limit, 0 else
*/
int has_critical_memory_demand(int total_variables, int dimension) {
    float system_memory = 16.0;

    // number of bytes of the tensor
    float tensor_size_bytes = (pow(dimension, total_variables)) * sizeof(long);
    // size in GB of the tensor
    float tensor_size_gigabytes = tensor_size_bytes / float(pow(10, 9));

    // number of bytes of the mps
    float train_size_bytes = (total_variables*dimension*pow(dimension, 2)) * sizeof(long);
    // size in GB of the mps
    float train_size_gigabytes = train_size_bytes / float(pow(10, 9));

    // total size of mps
    float total_size = tensor_size_gigabytes + train_size_gigabytes;
        
    // return if system memory is sufficient
    if(total_size >= system_memory) {
        // size is greater than system memory
        return 1;
    } else {
        // size is smaller than system memory
        return 0;
    }
}

/**
 * Used to test the generation of hmms
*/
int test_hmm() {
    // test the model generation
    HMM model = generate_hmm(3, 4, 3);
    println("Generated HMM with hidden dimension 3 and four hidden variables of dimension 3:");
    println("-------------------------------------------------------------------------------");

    println("Transition matrix:");
    PrintData(model.transition);

    println("Emission tensor:");
    PrintData(model.emission_tensor);

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
    cout << "\n"; 

    return 0;
}