#ifndef __FORWARD_ALGORITHM_
#define __FORWARD_ALGORITHM_

#include "itensor/all.h"
#include "../hmm/hmm.h"
#include <iostream>
#include <vector>
#include <thread>

enum parallelization_opt
{
    sequential, parallel_evidence, parallel_contraction, both_parallel
};

itensor::ITensor forward_alg(HMM model, std::vector<int>* evidence, int length, model_mode mode, parallelization_opt parallel);

itensor::ITensor calculate_forward_message(HMM model, std::vector<int>* evidence, int timestep, model_mode mode, parallelization_opt parallel, double* emission_values);

double* calculate_emission_values_parallel(HMM model, std::vector<int>* evidence, int length);

void calculate_emission_values(HMM model, std::vector<int>* evidence, int length, int start, double* emission_values);

void collect_data_forward_algorithm(int min_rank, int max_rank, int min_dimension, int max_dimension, int length, int repetitions, model_mode mode, parallelization_opt parallel);

float arithmetic_mean(float data[], int n);

float standard_mean_error(float data[], int n);

std::string parallel_to_string(parallelization_opt parallel);

#endif