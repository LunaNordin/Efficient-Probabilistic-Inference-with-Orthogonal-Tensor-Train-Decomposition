#include "itensor/all.h"

using namespace itensor;

/**
 * Constructs odeco, symmetric (nxnx...xn) tensor from set of eigenvectors.
 * rank: rank of the constructed tensor
 * dim: dimension of vector indices
 * vec: array of orthogonal eigenvectors
 * return: constructed vector
*/
ITensor generate_symmetric(int rank, int dim, float **vec) {

    // create rank-1 Tensor to represent eigenvectors
    auto i = Index(dim);
    auto A_i = ITensor(i);
    // create tensor for output
    auto T = ITensor();

    // sum over all additive parts
    for(int n=1; n <= rank; n++) {
        // fill vector with values from array
        for(int o=1; o <= dim; o++) {
            A_i.set(i=o, vec[n-1][o-1]);
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

int main(){
    
    // construct set of three orthogonal vectors
    float *vec[3];
    float temp[3][3] = {{2, 4, 0}, {-4, 2, 0}, {0, 0, 5}};
    for (int i = 0; i < 3; ++i)
    {
        vec[i] = temp[i];
    }
    // construc odeco, symmetric tensor of rank 3
    auto S = generate_symmetric(3, 3, vec);
    PrintData(S);

    return 0;
}