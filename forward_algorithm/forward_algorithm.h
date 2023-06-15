#ifndef __FORWARD_ALGORITHM_
#define __FORWARD_ALGORITHM_

#include "itensor/all.h"
#include "../hmm/hmm.h"

itensor::ITensor forward_alg_tensor(HMM model, std::vector<int>* evidence, int length);

itensor::ITensor calculate_forward_message(HMM model, std::vector<int>* evidence, int timestep);

#endif