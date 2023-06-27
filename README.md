# BA Frederik Schittny



## Setup and installation

In order to use the scripts provided in this repository you have to install the ITensor C++ library. To avoid platform specific problems you have to compile the library yourself. To do so follow the instructions of the [ITensor C++ installation guide](https://www.itensor.org/docs.cgi?vers=cppv3&page=install). Create your "itensor" library folder in the same directory you cloned this repo in. Your file structure should look like this:
    .  
    |─ ...  
    |─ project_folder  
    │ &emsp;&emsp;&emsp;  |─ itensor   
    │ &emsp;&emsp;&emsp;  │ &emsp;&emsp;&emsp;  |─ itensor   
    │ &emsp;&emsp;&emsp; │  &emsp;&emsp;&emsp; |─ lib   
    │ &emsp;&emsp;&emsp;  │  &emsp;&emsp;&emsp; |─ tools  
    │ &emsp;&emsp;&emsp;  │  &emsp;&emsp;&nbsp;&nbsp; └─ ...  
    │ &emsp;&emsp;&emsp;  |─ gitlab_repo   
    │ &emsp;&emsp;&emsp;  │  &emsp;&emsp;&emsp; |─ eigen  
    │ &emsp;&emsp;&emsp; |  &emsp;&emsp;&emsp;  |─ symmetric_tensor  
    │ &emsp;&emsp;&emsp; |  &emsp;&emsp;&nbsp;&nbsp;&nbsp;└─...   
    │ &emsp;&emsp;&nbsp;&nbsp; └─...   

Actually ... never mind. Dependencies in this repo are kind of cursed right now. Will fix soon hopefully...