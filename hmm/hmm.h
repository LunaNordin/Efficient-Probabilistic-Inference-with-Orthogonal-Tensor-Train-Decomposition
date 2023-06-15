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

HMM generate_hmm(int hiddenDim, int visibleVariables, int visibleDim);

std::vector<int> generate_state(int visibleVariables, int visibleDim);

std::vector<int>* generate_state_sequence(int visibleVariables, int visibleDim, int length);

int test_hmm();

#endif