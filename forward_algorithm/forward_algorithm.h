#ifndef __FORWARD_ALGORITHM_
#define __FORWARD_ALGORITHM_

#include "itensor/all.h"
#include "../hmm/hmm.h"
#include <iostream>
#include <vector>

itensor::ITensor forward_alg_tensor(HMM model, std::vector<int>* evidence, int length);

itensor::ITensor calculate_forward_message(HMM model, std::vector<int>* evidence, int timestep);

void collect_data_forward_algorithm(int min_rank, int max_rank, int min_dimension, int max_dimension, int length, int repetitions);

float arithmetic_mean(float data[], int n);

float standard_mean_error(float data[], int n);

#endif