#ifndef __HMM_H_
#define __HMM_H_

#include "itensor/all.h"
#include "../symmetric_tensor/generate_symmetric.h"
#include <random>
#include <vector>
#include <iostream>
#include <thread>

class HMM {
  public:
    int visibleVariables;                             // number of visible variables
    int hiddenVariables;                              // number of hidden variables (NOTE: alway 1 for now)   
    int hiddenDimension;                              // dimension of hidden variable
    int visibleDimension;                             // dimension of all visible variables (NOTE: equal to hiddenDimension for now)
    itensor::MPS emission_mps;                        // MPS representation containing all emission probabilities
    itensor::ITensor emission_tensor;                 // tensor representation containing all emission probabilities
    itensor::ITensor transition;                      // matrix containing all transition probabilities
    std::vector<int> initial_state;                   // vector containing initial states of the visible variables
    itensor::ITensor initial_hidden_probability;      // rank-1 tensor containing the initial probabilities of the hidden states
};

enum model_mode
{
    tensor, mps, both_models
};

enum parallelization_mode
{
    sequential, parallel_evidence, parallel_contraction, both_parallel
};

class ParallelizationOpt {
    public:
    parallelization_mode mode;          // description of which parallelization options are utilised
    int parallel_evidence_threads;      // number of threads for parallel emission calculation

    // constructor specifying only the paralleization mode but no specific options
    ParallelizationOpt(parallelization_mode mode) {
        this->mode = mode;  // set the given mode
        // all other parameters are disabled by default
        parallel_evidence_threads = 0;

        // if the mode is not sequential the parameters for the parallelization have to be added
        // otherwise there will likely be no actual parallelization in the calculations
        if(mode == parallel_evidence || mode == both_parallel) {
            itensor::println("Warning: Non-sequential parallelization option without specified parallelization parameter.");
        }
    }

    // constructor specifying the mode and all necessary parameters
    ParallelizationOpt(parallelization_mode mode, int threads) {
        this->mode = mode;
        parallel_evidence_threads = threads;
    }
};

HMM generate_hmm(int hiddenDim, int visibleVariables, int visibleDim, model_mode mode);

std::vector<int> generate_state(int visibleVariables, int visibleDim);

std::vector<int>* generate_state_sequence(int visibleVariables, int visibleDim, int length);

int has_critical_memory_demand(int total_variables, int dimension, model_mode mode);

itensor::Real get_component_from_tensor_train(itensor::MPS train, std::vector<int> evidence, ParallelizationOpt parallel);

itensor::Real get_component_from_tensor_train_with_ckeck(HMM model, std::vector<int> evidence, ParallelizationOpt parallel);

void absorb_evidence(itensor::MPS& train, std::vector<int> evidence, int length, int start);

std::string mode_to_string(model_mode mode);

int test_hmm();

#endif