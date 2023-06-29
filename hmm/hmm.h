#ifndef __HMM_H_
#define __HMM_H_

#include "itensor/all.h"
#include "../symmetric_tensor/generate_symmetric.h"
#include <random>
#include <vector>
#include <iostream>


class HMM {
  public:
    int visibleVariables;
    int hiddenVariables;
    int hiddenDimension;
    int visibleDimension;
    itensor::MPS emission_mps;
    itensor::ITensor emission_tensor;
    itensor::ITensor transition;
    std::vector<int> initial_state;
    itensor::ITensor initial_hidden_probability;
};

enum model_mode
{
    tensor, mps, both
};

HMM generate_hmm(int hiddenDim, int visibleVariables, int visibleDim, model_mode mode);

std::vector<int> generate_state(int visibleVariables, int visibleDim);

std::vector<int>* generate_state_sequence(int visibleVariables, int visibleDim, int length);

int has_critical_memory_demand(int total_variables, int dimension, model_mode mode);

itensor::Real get_component_from_tensor_train(itensor::MPS train, std::vector<int> evidence);

itensor::Real get_component_from_tensor_train_with_ckeck(HMM model, std::vector<int> evidence);

std::string mode_to_string(model_mode mode);

int test_hmm();

#endif