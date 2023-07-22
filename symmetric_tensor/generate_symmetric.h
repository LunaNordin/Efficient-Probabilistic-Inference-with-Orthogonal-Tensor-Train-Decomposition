#ifndef __GENERATE_SYMMETRIC_H_
#define __GENERATE_SYMMETRIC_H_

#include "../itensor/itensor/all.h"
#include <Eigen/Dense>

Eigen::MatrixXf generate_orthogonal_set(int n, int m);

itensor::ITensor generate_symmetric_odeco_tensor(int rank, int dim, Eigen::MatrixXf eigenvec);

itensor::MPS generate_symmetric_odeco_tensor_train(int rank, int dim);

itensor::ITensor contract_tensor_train_parallel(itensor::MPS train, int length, int start);

itensor::ITensor contract_tensor_train(itensor::MPS train);

int test_generation();

#endif