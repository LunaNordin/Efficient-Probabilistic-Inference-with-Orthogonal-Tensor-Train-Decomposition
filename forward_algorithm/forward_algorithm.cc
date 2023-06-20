#include "forward_algorithm.h"

using namespace itensor;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::milliseconds;

/**
 * Wrapper funtion for starting a recursive forward algorithm run
 * model: model to perform the forward algorithm on
 * evidence: sequence of states
 * legth: length of the state sequence
 * return: a posteriori probabilities for the given evidence
*/
ITensor forward_alg_tensor(HMM model, vector<int>* evidence, int length) {
    
    ITensor joint_hidden_probability = calculate_forward_message(model, evidence, length);

    return joint_hidden_probability;
}

/**
 * Recursively calculates the state probabilities of the hidden variable for the given evidence using forward messages
 * model: model to perform the forward algorithm on
 * evidence: sequence of states
 * timestep: current step in the evidence sequence
 * return: forward message of the current timestep
*/
ITensor calculate_forward_message(HMM model, vector<int>* evidence, int timestep) {

    // TODO: check that alpha_prior has dimension of hidden var

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
    ITensor alpha_prior = calculate_forward_message(model, evidence, timestep - 1);
    
    // calculate the forward message for all possible values of the hidden variable
    for(int hidden_index = 1; hidden_index <= model.hiddenDimension; hidden_index++) {
        // add the index of the hidden state at the front of the index list
        timestep_indices.at(0) = hidden_index;

        // get the emission probability for the current visible state and hidden state
        emission_value = elt(model.emission_tensor, timestep_indices);

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
}

/**
 * Automatically collects runtime data for the different algorithms with varying parameters and saves data to files
 * min_rank: lowest investigated rank
 * max_rank: highest investigated rank
 * min_dimension: lowest investigated dimension
 * max_dimension: highest investigated dimension
 * length: evidence sequence timesteps
 * repetitions: number of repetitions with identical parameters
*/
void collect_data_forward_algorithm(int min_rank, int max_rank, int min_dimension, int max_dimension, int length, int repetitions) {

    // variables used during testing
    HMM model;
    vector<int>* evidence;
    ITensor a_posteriori_probabilities;

    // create a new output files or open the existing ones
    ofstream fout_results;
    fout_results.open("forward_algorithm_recursive_tensor.csv", ios::out | ios::app);

    ofstream fout_error;
    fout_error.open("forward_algorithm_recursive_tensor_error.csv", ios::out | ios::app);

    ofstream fout_rel_error;
    fout_rel_error.open("forward_algorithm_recursive_tensor_rel_error.csv", ios::out | ios::app);

    // write test parameters to files
    fout_results << "min_rank:" << min_rank << ",max_rank:" << max_rank << ",min_dimension:" << min_dimension
    << ",max_dimension:" << max_dimension << ",length:" << length << ",repetitions:" << repetitions << "\n";
    fout_error << "min_rank:" << min_rank << ",max_rank:" << max_rank << ",min_dimension:" << min_dimension
    << ",max_dimension:" << max_dimension << ",length:" << length << ",repetitions:" << repetitions << "\n";
    fout_rel_error << "min_rank:" << min_rank << ",max_rank:" << max_rank << ",min_dimension:" << min_dimension
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

    // go through every possible combination of rank and dimension
    for(int rank = min_rank; rank <= max_rank; rank++) {

        fout_results << rank << ",";
        fout_error << rank << ",";
        fout_rel_error << rank << ",";

        cout << "Measuring runtime for Tensors of rank: " << rank << endl; 

        // test all possible dimensions for the current rank
        for(int dimension = min_dimension; dimension <= max_dimension; dimension++) {
            
            // if the combination of rank and dimension needs too much memors start testing the next rank
            if(has_critical_memory_demand(rank, dimension)) {
                fout_results << "\n";
                fout_error << "\n";
                fout_rel_error << "\n";
                break;
            }

            // array to save the result of each run
            float results[repetitions];

            // repeat measurement with identical parameters
            for(int i = 0; i < repetitions; i++) {
                // generate a new model and an evidence sequence
                model = generate_hmm(dimension, rank-1, dimension);
                evidence = generate_state_sequence(model.visibleVariables, model.visibleDimension, length);

                // perform the algortihm while measuring the runtime
                auto t1 = high_resolution_clock::now();
                a_posteriori_probabilities = forward_alg_tensor(model, evidence, length);
                auto t2 = high_resolution_clock::now();

                // save the measured result
                duration<double, std::milli> ms_double = t2 - t1;
                results[i] = ms_double.count();

                // clear all data structures of this run to make sure memory limit wont be exceeded
                delete[] evidence;
                model = HMM();
                a_posteriori_probabilities = ITensor();
                
                // experimental: give the system some time to clear the deallocated memory while this thread sleeps
                sleep(1);
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

int main() {

    collect_data_forward_algorithm(4, 10, 2, 100, 500, 10);

    // HMM model = generate_hmm(20, 4, 20);
    // int length = 100;
    // vector<int>* evidence = generate_state_sequence(model.visibleVariables, model.visibleDimension, length);

    // println("Initial hidden probabilities:");
    // PrintData(model.initial_hidden_probability);

    // println("Random state sequence:");
    // for (int i = 0; i < length; i++) {
    //     copy(evidence[i].begin(), evidence[i].end(), ostream_iterator<int>(cout, " "));
    //     cout << "  "; 
    // }
    // cout << "\n\n";

    // auto t1 = high_resolution_clock::now();
    // ITensor a_posteriori_probabilities = forward_alg_tensor(model, evidence, length);
    // auto t2 = high_resolution_clock::now();

    // duration<double, std::milli> ms_double = t2 - t1;
    // std::cout << ms_double.count() << "ms\n";

    // delete[] evidence;

    // println("A posteriori probabilities:");
    // PrintData(a_posteriori_probabilities);

    return 0;
}