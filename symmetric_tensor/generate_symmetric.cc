#include "generate_symmetric.h"

using namespace itensor;
using Eigen::MatrixXf;

/**
 * Constructs odeco, symmetric (nxnx...xn) tensor from set of orthogonal eigenvectors.
 * rank: rank of the constructed tensor
 * dim: dimension of vector indices
 * eigenvec: matrix with eigenvectors as columns
 * return: constructed vector
*/
ITensor generate_symmetric_odeco_tensor(int rank, int dim, MatrixXf eigenvec) {

    // create rank-1 Tensor to represent eigenvectors
    auto i = Index(dim);
    auto A_i = ITensor(i);
    // create tensor for output
    auto T = ITensor();

    // TODO: make sure the matrix of eigenvectors fits the rank and dimension of the generated vector

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

    // add tags to mark indices which will bond carriages inside tensor train and visible indices
    auto a = i;
    // there have to be two indices to bond to the carriages left and right
    T = addTags(T,"bond1", a);
    a = prime(a);
    T = addTags(T,"bond2", a);
    // add a "visible" tag to all remaining indices
    for(int p=1; p <= rank-1; p++) {
        a = prime(a);
        // add a number to each visible indice to distinguish them
        std::string tag = "visible";
        tag += std::to_string(p);
        T = addTags(T, tag, a);
    }

    // remove unused prime for better readablilty (indices are now distinguished by tags)
    T = noPrime(T);

    // return the output
    return T;
}

/**
 * Generates set of orthogonal vectors with specified dimesion using houser QR decomposition
 * n: number of rows (dimension of generated orthogonal vectors)
 * m: number of columns (number of generated orthogonal vectors)
 * return: matrix with m orthogonal, n-dimensional vectors as columns
*/
MatrixXf generate_orthogonal_set(int n, int m) {
    
    // generate a random nxm matrix
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

    // TODO: add multiplication with random factors to generate tensors with values outside [0,1]

    // shift from generated range of [-1,1] to [0,1] for working with probabilities
    Q = Q.cwiseAbs();

    // return the genrated matrix
    return Q;
}

/**
 * Generates a symmetric, odeco tansor train (MPS)
 * rank: rank of the tensor represented by the mps (number of carriages - 2 due to two visible indices at the ends)
 * dim: dimension of the indices of the represented tensor (and all carriages)
 * return: mps representing a tensor with specified rank and dimensions
*/
MPS generate_symmetric_odeco_tensor_train(int rank, int dim) {

    // iniitalize random number generation
    srand((unsigned int) time(0));

    // initialise an empty train for rank - 2 many carriages
    auto train = MPS(rank - 2);

    // indices used to connect the carriages of the train
    auto bond_1 = Index(dim);
    auto bond_2 = Index(dim);

    // generate each carriage
    for(int i=1; i <= rank - 2; i++) {
        // create an symmetric, odeco tensor
        MatrixXf base = generate_orthogonal_set(dim, rank - 2);
        auto carriage = generate_symmetric_odeco_tensor(3, dim, base);

        // find the indices of the carriages marked as bonds
        auto old_bond_1 = findIndex(carriage,"bond1");
        auto old_bond_2 = findIndex(carriage,"bond2");

        // replace the indices with new ones which are not identical and untagged
        // this makes sure each carriage shares one index with each of its neighbours
        carriage = replaceInds(carriage,{old_bond_1, old_bond_2},{bond_1, bond_2});
        
        // the carriages at the end of the train only share one bond but have two visible indices
        if(i == 1) {
            carriage = addTags(carriage,"visible2",bond_1);
        } else if (i == rank - 2){
           carriage = addTags(carriage,"visible2",bond_2);
        }
        
        // the right bond of this carriage has to become the left bond of the next carriage
        bond_1 = bond_2;
        // create a new right bond for the next carriage
        bond_2 = Index(dim);

        // add the generated carriage to the train
        train.set(i, carriage);
    }
    // return the generated train
    return train;
}

/**
 * Contracts a tensor train into the represented tensor
 * train: tensor train to be contracted
 * return: tensor represented by the tensor train
*/
ITensor contract_tensor_train(MPS train) {

    // set output tensor to first tensor in mps
    auto T = train.ref(1);;

    // contract all carriages
    for(int i = 2; i <= train.length(); i++) {
        // get the next carriage
        auto S = train.ref(i);

        // automatically contract over the shared index
        T *= S;
    }
    // remove all tags (all indices are visible now)
    T = noTags(T);
    // return the contracted tensor
    return T;
}

/**
 * Used to test the generation of symmetric tensors and tensor trains
*/
int test_generation(){

    // initialize seed for random number generation
    srand((unsigned int) time(0));

    // println("Test with known eigenvectors:");
    // /**
    //  * Test algorithm with matrix of eigenvectors:
    //  * 2 -4  0
    //  * 4  2  0
    //  * 0  0  5
    //  * Result for this test is known.
    // */
    // MatrixXf M1 = MatrixXf(3,3);
    // M1(0,0) = 2;
    // M1(1,0) = 4;
    // M1(2,0) = 0;
    // M1(0,1) = -4;
    // M1(1,1) = 2;
    // M1(2,1) = 0;
    // M1(0,2) = 0;
    // M1(1,2) = 0;
    // M1(2,2) = 5;
    // println(M1);
    // auto S1 = generate_symmetric_odeco_tensor(3, 3, M1);
    // PrintData(S1);

    // println("============================================");

    // println("Test with unknown eigenvectors:");
    // MatrixXf M2 = generate_orthogonal_set(3, 3);
    // println(M2);
    // auto S2 = generate_symmetric_odeco_tensor(3, 3, M2);
    // PrintData(S2);

    // println("============================================");

    println("Generate a symmetric tensor train:");
    auto train = generate_symmetric_odeco_tensor_train(4, 6);
    PrintData(train);

    println("Represented tensor:");
    auto T = contract_tensor_train(train);
    PrintData(T);

    return 0;
}