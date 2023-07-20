#ifndef __FORWARD_ALGORITHM_
#define __FORWARD_ALGORITHM_

#include "itensor/all.h"
#include "../hmm/hmm.h"
#include <iostream>
#include <vector>
#include <thread>

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
        if(mode != sequential) {
            itensor::println("Warning: Non-sequential parallelization option without specified parallelization parameter.");
        }
    }

    // constructor specifying the mode and all necessary parameters
    ParallelizationOpt(parallelization_mode mode, int threads) {
        this->mode = mode;
        parallel_evidence_threads = threads;
    }
};

itensor::ITensor forward_alg(HMM model, std::vector<int>* evidence, int length, model_mode mode, ParallelizationOpt parallel);

itensor::ITensor calculate_forward_message(HMM model, std::vector<int>* evidence, int timestep, model_mode mode, ParallelizationOpt parallel, double* emission_values);

double* calculate_emission_values_parallel(HMM model, std::vector<int>* evidence, int length, int num_threads);

void calculate_emission_values(HMM model, std::vector<int>* evidence, int length, int start, double* emission_values);

void collect_data_forward_algorithm(int min_rank, int max_rank, int min_dimension, int max_dimension, int length, int repetitions, model_mode mode, ParallelizationOpt parallel);

float arithmetic_mean(float data[], int n);

float standard_mean_error(float data[], int n);

std::string parallel_to_string(ParallelizationOpt parallel);

#endif