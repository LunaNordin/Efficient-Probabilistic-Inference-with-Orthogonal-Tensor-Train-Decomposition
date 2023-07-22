#ifndef __FORWARD_ALGORITHM_
#define __FORWARD_ALGORITHM_

#include "../itensor/itensor/all.h"
#include "../hmm/hmm.h"
#include <iostream>
#include <vector>
#include <thread>

itensor::ITensor forward_alg(HMM model, std::vector<int>* evidence, int length, model_mode mode, ParallelizationOpt parallel);

itensor::ITensor calculate_forward_message(HMM model, std::vector<int>* evidence, int timestep, model_mode mode, ParallelizationOpt parallel, double* emission_values);

double* calculate_emission_values_parallel(HMM model, std::vector<int>* evidence, int length, ParallelizationOpt parallel);

void calculate_emission_values(HMM model, std::vector<int>* evidence, int length, int start, double* emission_values, ParallelizationOpt parallel);

void collect_data_forward_algorithm(int min_rank, int max_rank, int min_dimension, int max_dimension, int length, int repetitions, model_mode mode, ParallelizationOpt parallel);

float arithmetic_mean(float data[], int n);

float standard_mean_error(float data[], int n);

std::string parallel_to_string(ParallelizationOpt parallel);

#endif