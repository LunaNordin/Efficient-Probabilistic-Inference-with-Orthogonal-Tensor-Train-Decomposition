
# BA Frederik Schittny

## Setup and installation

In order to use the scripts provided in this repository you have to install the ITensor and Eigen C++ libraries. To avoid platform specific problems you have to add the libraries yourself. To do so follow the instructions in the [ITensor C++ installation guide](https://www.itensor.org/docs.cgi?vers=cppv3&page=install) and the [Eigen documentation](https://eigen.tuxfamily.org/dox/GettingStarted.html) . It is advisable to build the itensor library yourself to choose optimal setting for your systems compiler and system specific accelerations.

Install both libraries in their own directories named "itensor" and "eigen" respectively within the git repositories root directory. The final file structure should look like this:
    .  
    |─ ...  
    |─ gitlab_folder  
    │ &emsp;&emsp;&emsp;  |─ data_analysis   
    │ &emsp;&emsp;&emsp;  |─ eigen   
    │ &emsp;&emsp;&emsp;  |─ forward_algorithm  
    │ &emsp;&emsp;&emsp;  |─ hmm      
    │ &emsp;&emsp;&emsp;  |─ itensor   
    │ &emsp;&emsp;&emsp;  |─ lapack_version  
    │ &emsp;&emsp;&emsp;  |─ symmetric_tensor      
    │ &emsp;&emsp;&nbsp;&nbsp; └─...   

The main .gitignore file of this repo will prevent that any contents of the dependencies will be commited to the repository. If you want to install the dependencies in another location or happen to already have working installs please note, that you will have to adjust the references in the makefiles as well as the header file imports yourself.

The project also depends on a system wide installation of LAPACK. To test whether your system already has a working install,  compile and run the C-script in /lapack_version. The minimum required version is 3.2.1. If you need to install LAPACK on your system please follow the instructions on the [LAPACK website](https://netlib.org/lapack/) .

To test your dependencies you can compile and run one of the provided demos located in /itensor/tutorial and /eigen/demos.