

# Efficient Probabilistic Inference with Orthogonal Tensor Train Decomposition
This repository contains the implementation used to gather runtime data as well as the data analysis for the accompanying bachelor thesis paper "Efficient Probabilistic Inference with Orthogonal Tensor Train Decomposition" written by Frederik Schittny. This readme file explains how to setup the implementation and how to execute the code to recreate the runtime measurements describe in the thesis.

## Setup and installation
In order to use the scripts provided in this repository you have to install the ITensor and Eigen C++ libraries after you cloned this repository. To avoid platform specific problems you have to add the libraries yourself. The project structure allready includes the folders "itensor" and "eigen" for the dependencies, which only have to be filled with the dependency files:.  
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

In order to install the Eigen library follow the instructions in the [Eigen documentation](https://eigen.tuxfamily.org/dox/GettingStarted.html) and place the library files into the prepared folder named "eigen". The ITensor library is included as a submodule in this repository. Please use the command "git submodule update --init --recursive" to obtain the version of ITensor which is used in the thesis paper. After that, follow the instructions from the [ITensor installation guide](https://itensor.org/docs.cgi?vers=cppv3&page=install) to adjust and compile the ITensor files to your machine and OS.

The main .gitignore file of this repo will prevent that any contents of the dependencies will be commited to the repository. If you want to install the dependencies in another location or happen to already have working installs please note that you will have to adjust the references in the makefiles as well as the header file imports yourself.

The project also depends on a system wide installation of LAPACK. To test whether your system already has a working install compile and run the C-script in the "lapack_version" folder. The minimum required version is 3.2.1. If you need to install LAPACK on your system please follow the instructions on the [LAPACK website](https://netlib.org/lapack/).

To test your dependencies you can compile and run one of the provided demos located in /itensor/tutorial and /eigen/demos. The complete setup of the repository can be tested by running the "make" command in the "forward_algortihm" folder and executing the compiled binary. In the case of a successfull installation this should execute the validation method of the installation and printout the following to the console before returning:

Checking similarity of implementation results for rank 4  
Checking similarity of implementation results for rank 5  
Checking similarity of implementation results for rank 6  

## Runtime Measurements
By default, the main method in the forward_algorithm file only executes code to validate the different implementations of the forward algortihm. To recreate the measurements discussed in the thesis you can comment out the lines starting with "collect&#95;data&#95;forward&#95;algorithm" under the "Evidence sequence length 400" comment. Please note that some of the measurements took over 24 hours to complete on the hardware used for the thesis. Depending on the hardware you are using you might want to consider running the measurements one by one or using shorter evidence sequences to reduce the runtime of the forward algorithm. There are measurements with equivalent setups but shorter runtimes listed under the comment "Evidence sequence length 100". 

If you want to adapt or change the implementation you can use the measurements listed under "Quick testing with very short runtimes" to verify that measurements run at all or produce realistic results. There are also test methods to test the functionality of different parts of the implementation. For further implementation details please refer to the "Implementation" chapter of the accompanying thesis paper or the inline comments in the code. 

## Data analysis
The data analysis for the thesis was done mainly with [MagicPlot](https://magicplot.com/), which was also used to create the visualisations shown in the thesis. In order to open the .mppz files containing the calculations and analysis you have to install MagicPlot to your machine. The folder "data_analysis" does not only contain the MagicPlot files but also the runtime results in .csv format which the analysis is based on as well as the visualisations exported for use in the thesis. Please note that only the runtimes measured on a Mac mini were used for the thesis. All runtimes and analysis located in the folder for the MacBook Pro were recorded during the development of the implementation as a proof of concept and might not refelct runtimes which can be aqcuired with the current implementation state.