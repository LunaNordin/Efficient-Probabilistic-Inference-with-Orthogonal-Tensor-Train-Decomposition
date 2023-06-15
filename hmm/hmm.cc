#include "hmm.h"
#include "../symmetric_tensor/generate_symmetric.h"

using namespace itensor;

/**
 * Generates a hidden markov model with specified dimensions and random values.
 * hiddenDim: dimension of the one hidden variable
 * visibleVariables: number of visible variables
 * visibleDim: dimension of all visible variables
 * return: HMM object with specified number of varibales and dimensions as well as random values in transition, emision and initial state.
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

    // generate a symmetric, odeco tensor train as the mps representation of the emission tensor
    model.emission_mps = generate_symmetric_odeco_tensor_train(visibleVariables, visibleDim);
    // contract that tensor train and also save the contracted emission tensor
    model.emission_tensor = contract_tensor_train(model.emission_mps);

    // add the initial state to the model
    model.initial_state = generate_state(visibleVariables, visibleDim);

    // return the generated model
    return model;
}

/**
 * Generates a sequence of random states which can be used as evidence.
 * visibleVariables: Number of values in each state
 * visibleDim: value range of each state component
 * length: number of generated states
 * return: array of states in form of rank-1 ITensors
*/
ITensor* generate_state_sequence(int visibleVariables, int visibleDim, int length) {
    // set up an array of ITensors to hold the state sequence
    ITensor* sequence = new ITensor[length];

    // generate lenght many states
    for (int i = 1; i <= length; i++) {
        // generate a random state
        ITensor state = generate_state(visibleVariables, visibleDim);
        // add the state to the sequence
        sequence[i-1] = state;
    }
    // return the generated state sequence
    return sequence;
}

/**
 * Generates a random state for a model with specified parameters.
 * visibleVariables: number of variables for which there are states to be generated
 * visibleDim: dimension of the variables is range for number generation
 * return: ITensor of rank 1 and specified dimensions containing generated state
*/
ITensor generate_state(int visibleVariables, int visibleDim) {
    // set up a random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(1,visibleDim);

    // create a vector (rank-1 tensor) for the state
    auto l = Index(visibleVariables);
    auto state = ITensor(l);

    // fill the state vector with random values from the dimension value range
    for (int j = 1; j <= visibleVariables; j++) {
        auto random = uni(rng);
        state.set(l = j, random);
    }
    // return the generated state
    return state;
}

int main() {
    // test the model generation
    HMM model = generate_hmm(3, 4, 3);
    println("Generated HMM with hidden dimension 3 and four hidden variables of dimension 3:");
    println("-------------------------------------------------------------------------------");

    println("Transition matrix:");
    PrintData(model.transition);

    println("Emission tensor:");
    PrintData(model.emission_tensor);

    println("Initial state:");
    PrintData(model.initial_state);

    println("Random state sequence:");
    auto sequence = generate_state_sequence(model.visibleVariables, model.visibleDimension, 4);
    for (int i = 0; i < 4; i++) {
        PrintData(sequence[i]);
    }

    return 0;
}