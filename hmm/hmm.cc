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

    // set up a random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(1,visibleDim);
    // create a vector (rank-1 tensor) for the initial state
    auto l = Index(visibleVariables);
    auto state = ITensor(l);
    // fill the state vector with random values from the visible dimension value range
    for (int i = 1; i <= visibleVariables; i++) {
        auto random = uni(rng);
        state.set(l = i, random);
    }
    // add the initial state to the model
    model.initial_state = state;

    // return the generated model
    return model;
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

    return 0;
}