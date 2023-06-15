#ifndef __HMM_H_
#define __HMM_H_

#include "itensor/all.h"
#include <random>

class HMM {
  public:
    int visibleVariables;
    int hiddenVariables;
    int hiddenDimension;
    int visibleDimension;
    itensor::MPS emission_mps;
    itensor::ITensor emission_tensor;
    itensor::ITensor transition;
    itensor::ITensor initial_state;
};

HMM generate_hmm(int hiddenDim, int visibleVariables, int visibleDim);

itensor::ITensor generate_state(int visibleVariables, int visibleDim);

itensor::ITensor* generate_state_sequence(int visibleVariables, int visibleDim, int length);

#endif