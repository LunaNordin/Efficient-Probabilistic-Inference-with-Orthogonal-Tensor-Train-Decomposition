#include "forward_algorithm.h"
#include <vector>

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



int main() {

    HMM model = generate_hmm(20, 4, 20);
    int length = 100;
    vector<int>* evidence = generate_state_sequence(model.visibleVariables, model.visibleDimension, length);

    // println("Initial hidden probabilities:");
    // PrintData(model.initial_hidden_probability);

    // println("Random state sequence:");
    // for (int i = 0; i < length; i++) {
    //     copy(evidence[i].begin(), evidence[i].end(), ostream_iterator<int>(cout, " "));
    //     cout << "  "; 
    // }
    // cout << "\n\n";

    auto t1 = high_resolution_clock::now();
    ITensor a_posteriori_probabilities = forward_alg_tensor(model, evidence, length);
    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";

    delete[] evidence;

    // println("A posteriori probabilities:");
    // PrintData(a_posteriori_probabilities);

    return 0;
}