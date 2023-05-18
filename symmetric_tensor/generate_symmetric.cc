#include "itensor/all.h"
#include <Eigen/Dense>

using namespace itensor;
using Eigen::MatrixXf;

/**
 * Constructs odeco, symmetric (nxnx...xn) tensor from set of orthogonal eigenvectors.
 * rank: rank of the constructed tensor
 * dim: dimension of vector indices
 * eigenvec: matrix with eigenvectors as columns
 * return: constructed vector
*/
ITensor generate_symmetric_odeco(int rank, int dim, MatrixXf eigenvec) {

    // create rank-1 Tensor to represent eigenvectors
    auto i = Index(dim);
    auto A_i = ITensor(i);
    // create tensor for output
    auto T = ITensor();

    // sum over all additive parts
    for(int n=1; n <= rank; n++) {
        // get the n-th eigenvector from the matrix
        auto vec = eigenvec.col(n-1);
        // copy the values from the vector into the rank-1 tensor
        for(int o=1; o <= dim; o++) {
            A_i.set(i=o, vec(o-1)); // note that Eigen indices start at 0 while iTensor indices start at 1
        }

        // perfrom rank many tensor eigenproducts of eigenvector A_i
        auto B = A_i;
        for(int m=1; m < rank; m++) {
            B = B * prime(A_i,m);
        }
        
        // add the component to the output vector
        T += noPrime(B,i);
    }
    // return the output
    return T;
}

/**
 * Generates set of orthogonal vectors with specified dimesion using houser QR decomposition
 * n: number of rows (dimension of generated orthogonal verctors)
 * m: number of columns (number of generated orthogonal verctors)
 * return: matrix with m orthogonal, n-dimensional vectors as columns
*/
MatrixXf generate_orthogonal_set(int n, int m) {
    
    // generate a random 3x3 matrix
    srand((unsigned int) time(0));
    MatrixXf M = MatrixXf::Random(n,m);
    // perform QR decomposition using householder
    Eigen::FullPivHouseholderQR<MatrixXf> QR(M.rows(), M.cols());
    QR.compute(M);
    // get orthonormal matrix from decomposition
    MatrixXf Q = QR.matrixQ();
    // test matrix for orthogonality
    MatrixXf Q_t = Q.transpose();
    MatrixXf I = Q*Q_t;
    if (!I.isIdentity(1e-6)) {
        println("ERROR: Generated orthogonal matrix with high numeric error.");
        return MatrixXf();
    }
    // return the genrated matrix
    return Q;
}

int main(){

    println("Test with known eigenvectors:");
    /**
     * Test algorithm with matrix of eigenvectors:
     * 2 -4  0
     * 4  2  0
     * 0  0  5
     * Result for this test is known.
    */
    MatrixXf M1 = MatrixXf(3,3);
    M1(0,0) = 2;
    M1(1,0) = 4;
    M1(2,0) = 0;
    M1(0,1) = -4;
    M1(1,1) = 2;
    M1(2,1) = 0;
    M1(0,2) = 0;
    M1(1,2) = 0;
    M1(2,2) = 5;
    println(M1);
    auto S1 = generate_symmetric_odeco(3, 3, M1);
    PrintData(S1);

    println("============================================");

    println("Test with unknown eigenvectors:");
    MatrixXf M2 = generate_orthogonal_set(3, 3);
    println(M2);
    auto S2 = generate_symmetric_odeco(3, 3, M2);
    PrintData(S2);

    return 0;
}