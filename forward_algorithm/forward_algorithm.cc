#include "forward_algorithm.h"

using namespace itensor;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::milliseconds;

/**
 * Wrapper function for starting a recursive forward algorithm run
 * model: model to perform the forward algorithm on
 * evidence: sequence of states
 * length: length of the state sequence
 * mode: which representation of the emission tensor to use
 * parallel: options for parallelization
 * return: a posteriori probabilities for the given evidence
*/
ITensor forward_alg(HMM model, vector<int>* evidence, int length, model_mode mode, ParallelizationOpt parallel) {
    ITensor joint_hidden_probability;   // ITensor containing the calculated probability for the last timestep

    // decide what to do based on chosen representation and parallelization options
    if(mode == both_models) {
        // the algorithm always uses one specified model representation
        println("Error: Forward algorithm can only be executed on one representation at a time.");
        return ITensor();
    } else if(mode == tensor && parallel.mode != sequential) {
        // only calculations on mps representations can be parallelized
        println("Error: There is no parallelization available for tensor representation..");
        return ITensor();
    } else if(parallel.mode == sequential || parallel.mode == parallel_contraction){
        // use the implementation of the forward algorithm with integrated calculation/retrieval of emission values
        joint_hidden_probability = calculate_forward_message(model, evidence, length, mode, parallel, nullptr);
    } else if(mode == mps && parallel.mode == parallel_evidence) {
        // use the implementation of the forward algorithm with parallelized evidence sequence calculation upfront
        // pre-calculate the emission values
        double* emission_values = calculate_emission_values_parallel(model, evidence, length, parallel);
        // start the forward algorithm running on the pre-calculated emission values
        joint_hidden_probability = calculate_forward_message(model, evidence, length, mode, parallel, emission_values);
        // free the memory used for the array with emission values
        delete emission_values;
    } else {
        println("Info: Not yet implemented.");
        return ITensor();
    }

    return joint_hidden_probability;
}

/**
 * Recursively calculates the state probabilities of the hidden variable for the given evidence using forward messages
 * model: model to perform the forward algorithm on
 * evidence: sequence of states
 * timestep: current step in the evidence sequence
 * mode: which representation of the emission tensor to use
 * parallel: options for parallelization
 * emission_values: pre-calculated sequence of accessed emission tensor components
 * return: forward message of the current timestep
*/
ITensor calculate_forward_message(HMM model, vector<int>* evidence, int timestep, model_mode mode, ParallelizationOpt parallel, double* emission_values) {

    // make sure the provided arguments are consistent with each other
    if(parallel.mode == parallel_evidence && !emission_values) {
        // if the calculation of the evidence values is supposed to be parallel, the values have to be calculated upfront and provided as an array
        println("Error: Parallelized calculation of emission values has to be done upfront and passed to this method.");
        return ITensor();
    } else if(parallel.mode == sequential && emission_values) {
        // if the emission values are calculated on demand it does not make sense to provide pre-calculated values
        println("Warning: Pre-calculated emission values will not be used in this non-parallelized mode.");
    }

    auto i = Index(model.hiddenDimension);
    ITensor alpha_t = ITensor(i);           // array (rank-1 tensor) of forward messages which will be used in the next timestep (t+1)
    float emission_value;                   // value from the emission tensor for the current evidence and one hidden state
    float transition_value;                 // value from the transition matrix for current hidden state and prior hidden state

    // recursion anker
    if(timestep == 0) {
        // the forward messages for the initial timestep are the initial probabilities of the hidden variable
        return model.initial_hidden_probability;
    }

    // get the state of the visible variables for this timestep as an index list (all but the first index stay the same for this timestep)
    vector<int> timestep_indices = evidence[timestep - 1];
    // add one element at the front of the vector for the index of the hidden varible (will be replaced for each hidden state)
    timestep_indices.insert(timestep_indices.begin(), 0);

    // revursive function call to calculate forward messages of prior timestep
    ITensor alpha_prior = calculate_forward_message(model, evidence, timestep - 1, mode, parallel, emission_values);
    
    // calculate the forward message for all possible values of the hidden variable
    for(int hidden_index = 1; hidden_index <= model.hiddenDimension; hidden_index++) {
        // add the index of the hidden state at the front of the index list
        timestep_indices.at(0) = hidden_index;

        // get the emission probability for the current visible state and hidden state
        if(parallel.mode == sequential) {
            // the algorithm is not parallelized so get/calculate the emission value from the given model now

            if(mode == tensor) {
                // directly from the tensor representation
                emission_value = elt(model.emission_tensor, timestep_indices);
            } else if(mode == mps) {
                // calculated from the tensor train
                emission_value = get_component_from_tensor_train(model, timestep_indices, parallel);
            } else {
                // modes like 'both' do not make sense during calculation
                println("Error: Not a valid mode for forward message calculation.");
                return ITensor();
            }
        } else if(parallel.mode == parallel_evidence) {
            // the emission values have been pre-calculated in parallel and can be simply accessed
            emission_value = emission_values[(timestep - 1) * model.hiddenDimension + (hidden_index - 1)];
        } else if(parallel.mode == parallel_contraction) {
            // calculated from the tensor train (actually function call does not differ between sequential mps and mps with parallel contraction)
            emission_value = get_component_from_tensor_train(model, timestep_indices, parallel);
        } else {
            // other parallelization options are not yet implemented
            println("Error: Not a valid parallelization option for forward message calculation.");
            return ITensor();
        }

        // value for result of sub-calculation
        float sum = 0;

        // use the messages for the prior timestep to calculate new forward message
        for(int prior_message_index = 1; prior_message_index <= model.hiddenDimension; prior_message_index++) {
            
            // get transation value for current hidden state and prior hidden state
            transition_value = elt(model.transition, hidden_index, prior_message_index);
            // add value for this combination of current and prior state to the sum
            sum += transition_value * elt(alpha_prior, prior_message_index);
        }
        // calculate and add the forward message for the current hidden state to the array
        alpha_t.set(hidden_index, emission_value * sum);
    }
    // return the forward message for usage in the next timestep
    return alpha_t;
    // alternative to prevent "probabilities" of hidden states to go torwards 0 because the data is random and not actual probabilities
    // return alpha_t * 10;
}

/**
 * Splits up the given evidence sequence into subsequences and calculates the emission values in parallel in the order they will be accessed by the forward algorithmin
 * model: model to perform the forward algorithm on
 * evidence: sequence of states
 * length: length of the evidence sequence
 * parallel: options for parallelization
 * return: array of emission values in the order they will be accessed by the forward algorithm
*/
double* calculate_emission_values_parallel(HMM model, vector<int>* evidence, int length, ParallelizationOpt parallel) {

    int num_threads = parallel.parallel_evidence_threads;
    thread threads[num_threads];    // array containing the indiviual threads which will be working on parts of the evidence sequence
    double* emission_values = new double[length * model.hiddenDimension];   // array to store the calculated emission values

    // create a thread for each section of the evidence sequence
    for(int i = 0; i < num_threads; i++) {
        // calculate start, end and length of each section
        int start = (length * i) / num_threads;
        int end = ((length * (i + 1)) / num_threads) - 1;
        int length = end - start + 1;
        // create a thread to calculate all emission values for that part of the sequence
        threads[i] = thread(calculate_emission_values, model, evidence, length, start, emission_values, parallel);
        threads[i].join();
    }

    // wait until all threads have finished
    for(int i = 0; i < num_threads; i++) {
        // make this function call wait for the completion of this thread
        threads[i].join();
    }
    
    // return the calculated emission values
    return emission_values;
}

/**
 * Calculates all emission values in a range of an evidence sequence which will be accessed by the forward algorithm on the given model
 * model: model to perform the forward algorithm on
 * evidence: sequence of states
 * start: index in evidence sequence where to begin calculation
 * length: length of the state sequence for which the emission values are to be calculated
 * emission_values: array to be filled with emission values in the order they will be accessed by the forward algorithm
 * parallel: options for parallelization
*/
void calculate_emission_values(HMM model, vector<int>* evidence, int length, int start, double* emission_values, ParallelizationOpt parallel) {

    // make sure there is a model to calculate the values from
    if(!model.emission_mps) {
        println("Error: Model does not contain an mps representation to calculate emission values from.");
        return;
    }
    // make sure there is an array to write the calculated values to
    if(!emission_values) {
        println("Error: Invalid array to store emission values.");
        return;
    }

    int array_offest = 0;   // number of fields which have been filled from the starting index in the array

    // go thorugh the part of the evidence sequence assigned to this thread
    for(int i = 0; i < length; i++) {

        // get the state of the visible variables for this timestep as an index list (all but the first index stay the same for this timestep)
        vector<int> timestep_indices = evidence[start + i];
        // add one element at the front of the vector for the index of the hidden varible (will be replaced for each hidden state)
        timestep_indices.insert(timestep_indices.begin(), 0);

        // the forward algorithm needs the emission values of each hidden variable state for each evidence
        for(int hidden_index = 1; hidden_index <= model.hiddenDimension; hidden_index++) {
            // add the index of the hidden state at the front of the index list
            timestep_indices.at(0) = hidden_index;

            // debug printout to see which array values have been filled by this thread
            // cout << start * model.hiddenDimension + array_offest << endl;

            // calculate the emission value and add it to the correct field in the array (startindex + offset)
            // fill the part of the total array which corresponds to the part of the emission sequence calculated in this thread
            auto emission_value = get_component_from_tensor_train(model, timestep_indices, parallel);
            emission_values[start * model.hiddenDimension + array_offest] = emission_value;

            // fill the next array field in the next loop
            array_offest++;
        }
    }
}

/**
 * Automatically collects runtime data for the different algorithms with varying parameters and saves data to files
 * min_rank: lowest investigated rank
 * max_rank: highest investigated rank
 * min_dimension: lowest investigated dimension
 * max_dimension: highest investigated dimension
 * length: evidence sequence timesteps
 * repetitions: number of repetitions with identical parameters
 * mode: which representation of the emission tensor to use
 * parallel: options for parallelization
*/
void collect_data_forward_algorithm(int min_rank, int max_rank, int min_dimension, int max_dimension, int length, int repetitions, model_mode mode, ParallelizationOpt parallel) {

    // variables used during testing
    HMM model;
    vector<int>* evidence;
    ITensor a_posteriori_probabilities;

    // create a new output files or open the existing ones
    ofstream fout_results;
    string time_file_name = "forward_algorithm_recursive_" + parallel_to_string(parallel) + "_" + mode_to_string(mode) + "_" + to_string(length) + ".csv";
    fout_results.open(time_file_name, ios::out | ios::app);

    ofstream fout_error;
    auto error_file_name = "forward_algorithm_recursive_" + parallel_to_string(parallel) + "_" + mode_to_string(mode) + "_" + to_string(length) + "_error" + ".csv";
    fout_error.open(error_file_name, ios::out | ios::app);

    ofstream fout_rel_error;
    auto rel_error_file_name = "forward_algorithm_recursive_" + parallel_to_string(parallel) + "_" + mode_to_string(mode) + "_" + to_string(length) + "_rel_error" + ".csv";
    fout_rel_error.open(rel_error_file_name, ios::out | ios::app);

    // write test parameters to files
    fout_results << "model:" << mode_to_string(mode) << ",parallelization:" << parallel_to_string(parallel) << ",min_rank:" << min_rank << ",max_rank:" << max_rank << ",min_dimension:" << min_dimension
    << ",max_dimension:" << max_dimension << ",length:" << length << ",repetitions:" << repetitions << "\n";
    fout_error << "model:" << mode_to_string(mode) << ",parallelization:" << parallel_to_string(parallel) << ",min_rank:" << min_rank << ",max_rank:" << max_rank << ",min_dimension:" << min_dimension
    << ",max_dimension:" << max_dimension << ",length:" << length << ",repetitions:" << repetitions << "\n";
    fout_rel_error << "model:" << mode_to_string(mode) << ",parallelization:" << parallel_to_string(parallel) << ",min_rank:" << min_rank << ",max_rank:" << max_rank << ",min_dimension:" << min_dimension
    << ",max_dimension:" << max_dimension << ",length:" << length << ",repetitions:" << repetitions << "\n";

    // write header line with column titels for the dimensions to files
    fout_results << ",";
    fout_error << ",";
    fout_rel_error << ",";
    for(int k = min_dimension; k <= max_dimension; k++) {
        fout_results << k << ",";
        fout_error << k << ",";
        fout_rel_error << k << ",";
    }
    fout_results << "\n";
    fout_error << "\n";
    fout_rel_error << "\n";

    cout << "Runtime testing for HMM " << parallel_to_string(parallel) << " forward algorithm on " << mode_to_string(mode) << " representation with evidence sequences of length " 
    << to_string(length) << " and " << to_string(repetitions) << " repetitions per run" << endl; 

    // go through every possible combination of rank and dimension
    for(int rank = min_rank; rank <= max_rank; rank++) {

        fout_results << rank << ",";
        fout_error << rank << ",";
        fout_rel_error << rank << ",";

        cout << "Measuring runtime for tensors of rank: " << rank << endl; 

        // test all possible dimensions for the current rank
        for(int dimension = min_dimension; dimension <= max_dimension; dimension++) {
            
            // if the combination of rank and dimension needs too much memors start testing the next rank
            int memory_usage = has_critical_memory_demand(rank, dimension, mode);
            if(memory_usage == 2) {
                fout_results << "\n";
                fout_error << "\n";
                fout_rel_error << "\n";
                break;
            }

            // array to save the result of each run
            float results[repetitions];

            // repeat measurement with identical parameters
            for(int i = 0; i < repetitions; i++) {
                // generate a new model which only contains the tested representation
                model = generate_hmm(dimension, rank-1, dimension, mode);
                // generate a new evidence sequence
                evidence = generate_state_sequence(model.visibleVariables, model.visibleDimension, length);

                // perform the algortihm while measuring the runtime
                auto t1 = high_resolution_clock::now();
                a_posteriori_probabilities = forward_alg(model, evidence, length, mode, parallel);
                auto t2 = high_resolution_clock::now();

                // save the measured result
                duration<double, std::milli> ms_double = t2 - t1;
                results[i] = ms_double.count();

                // clear all data structures of this run to make sure memory limit wont be exceeded
                delete[] evidence;
                model = HMM();
                a_posteriori_probabilities = ITensor();
                
                // experimental: give the system some time to clear the deallocated memory while this thread sleeps
                if(memory_usage == 1) {
                    sleep(1);
                }
            }

            // calculate mean of results
            float mean = arithmetic_mean(results, repetitions);
            // claculate standard error of results
            float error = standard_mean_error(results, repetitions);
            // claculate relative error of results
            float rel_error = error / mean;

            // print results to console
            cout << "dimension: " << dimension << " mean: " << mean << " error: " << error << " rel. error: " << rel_error << endl;

            // write the measured runtime to the files
            fout_results << mean << "," << flush;
            fout_error << error << "," << flush;
            fout_rel_error << rel_error << "," << flush;
        }
        // start the next line for the next rank
        fout_results << "\n";
        fout_error << "\n";
        fout_rel_error << "\n";
    }
    // in case the program runs twice without the files being reset the new values will just be written underneith
    fout_results << "\n";
    fout_error << "\n";
    fout_rel_error << "\n";
    // close the files
    fout_results.close();
    fout_error.close();
    fout_rel_error.close();

    cout << "\n" << endl; 
}

/**
 * Calculates the arithmetic mean of the given data sequence
 * data: array with values
 * n: number of values
 * return: arithmetic mean of data sequence
*/
float arithmetic_mean(float data[], int n) {
    // loop to calculate sum of array elements.
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum = sum + data[i];
    }
     
    // return calculated mean
    return sum / n;
}

/**
 * Calculates standard error for the arithmetic mean of the given data sequence
 * data: array with values
 * n: number of values
 * return: standard error for arithmetic mean
*/
float standard_mean_error(float data[], int n) {
    float sum = 0;
    // calculate mean
    float mean = arithmetic_mean(data, n);
    // calculate standard deviation
    for (int i = 0; i < n; i++) {
        sum = sum + pow((data[i] - mean),2);
    }
    // calculate and return standard error
    sum = sum * (1/(float(n)-1));
    return sqrt(sum);
}

/**
 * Gives a string represenation for any parallelization_mode value
 * parallel: options to be converted to string
 * return: options as a string
*/
string parallel_to_string(ParallelizationOpt parallel)
{
    // TODO: replace this, its a maintenance nightmare
    switch (parallel.mode)
    {
        case sequential:                return "sequential";
        case parallel_evidence:         return "parallel_evidence_" + to_string(parallel.parallel_evidence_threads) + "_threads";
        case parallel_contraction:      return "parallel_contraction";
        case both_parallel:             return "both_parallel";
        default:                        return "unknown parallelization_mode";
    }
}

int main() {

    /**
     * Thourough testing to gather data for analysis
    */
    // rank 4-10, dimension: 2-100, sequence_length: 100, repetitions: 10, used_model: tensor, no_parallelization
    // collect_data_forward_algorithm(4, 10, 2, 100, 100, 10, tensor, ParallelizationOpt(sequential));
    // rank 4-10, dimension: 2-100, sequence_length: 500, repetitions: 10, used_model: tensor, no_parallelization
    // collect_data_forward_algorithm(4, 10, 2, 100, 500, 10, tensor, ParallelizationOpt(sequential));

    // rank 4-10, dimension: 2-100, sequence_length: 100, repetitions: 10, used_model: tensor train, no_parallelization
    // collect_data_forward_algorithm(4, 10, 2, 100, 100, 10, mps, ParallelizationOpt(sequential));
    // rank 4-10, dimension: 2-100, sequence_length: 500, repetitions: 10, used_model: tensor train, no_parallelization
    // collect_data_forward_algorithm(4, 10, 2, 100, 500, 10, mps, ParallelizationOpt(sequential));

    // rank 4-10, dimension: 2-100, sequence_length: 100, repetitions: 10, used_model: tensor train, parallel_evidence: 2 threads
    // collect_data_forward_algorithm(4, 10, 2, 100, 100, 10, mps, ParallelizationOpt(parallel_evidence, 2));
    // rank 4-10, dimension: 2-100, sequence_length: 500, repetitions: 10, used_model: tensor train, parallel_evidence: 2 threads
    // collect_data_forward_algorithm(4, 10, 2, 100, 500, 10, mps, ParallelizationOpt(parallel_evidence, 2));
    // rank 4-10, dimension: 2-100, sequence_length: 100, repetitions: 10, used_model: tensor train, parallel_evidence: 4 threads
    // collect_data_forward_algorithm(4, 10, 2, 100, 100, 10, mps, ParallelizationOpt(parallel_evidence, 4));
    // rank 4-10, dimension: 2-100, sequence_length: 500, repetitions: 10, used_model: tensor train, parallel_evidence: 4 threads
    // collect_data_forward_algorithm(4, 10, 2, 100, 500, 10, mps, ParallelizationOpt(parallel_evidence, 4));
    // rank 4-10, dimension: 2-100, sequence_length: 100, repetitions: 10, used_model: tensor train, parallel_evidence: 8 threads
    // collect_data_forward_algorithm(4, 10, 2, 100, 100, 10, mps, ParallelizationOpt(parallel_evidence, 8));
    // rank 4-10, dimension: 2-100, sequence_length: 500, repetitions: 10, used_model: tensor train, parallel_evidence: 8 threads
    // collect_data_forward_algorithm(4, 10, 2, 100, 500, 10, mps, ParallelizationOpt(parallel_evidence, 8));

    /**
     * Quick testing to try stuff out
    */
    // rank 4-5, dimension: 2-10, sequence_length: 10, repetitions: 3, used_model: tensor, no_parallelization
    // collect_data_forward_algorithm(4, 5, 2, 10, 10, 3, tensor, ParallelizationOpt(sequential));

    // rank 4-5, dimension: 2-10, sequence_length: 10, repetitions: 3, used_model: tensor train, no_parallelization
    // collect_data_forward_algorithm(4, 5, 2, 10, 10, 3, mps, ParallelizationOpt(sequential));

    // rank 4-5, dimension: 2-10, sequence_length: 10, repetitions: 3, used_model: tensor train, parallel_evidence: 2 threads
    // collect_data_forward_algorithm(4, 5, 2, 10, 10, 3, mps, ParallelizationOpt(parallel_evidence, 2));

    // rank 4-5, dimension: 2-10, sequence_length: 10, repetitions: 3, used_model: tensor train, parallel_contraction
    collect_data_forward_algorithm(4, 5, 2, 10, 10, 3, mps, ParallelizationOpt(parallel_contraction));

    // test_generation();
    // test_hmm();

    return 0;
}