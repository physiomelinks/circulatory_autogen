# Getting Started

## Initialising and Startup

**1. Install OpenCOR**

Download and install OpenCOR version 0.8.1 from this [link](https://opencor.ws/downloads/index.html). I recommend installing with zip/tarball in a directory where you have access and edit rights, such as ~/Desktop.

!!! note
    If you are not familiar with OpenCOR, you should go through the OpenCOR Tutorial before starting this

    Download the OpenCOR tutorial, which is a comprehensive tutorial including many examples: [OpenCOR Tutorial](https://tutorial-on-cellml-opencor-and-pmr.readthedocs.io/en/latest/_downloads/d271cfcef7e288704c61320e64d77e2d/OpenCOR-Tutorial-v17.pdf).

**2. Clone the project**

Clone the Circulatory Autogen project from the [GitHub repository](https://github.com/FinbarArgus/circulatory_autogen).

!!! note
    If you have not worked with git and GitHub, firstly download and install git, and then open the terminal and navigate (with terminal in Linux/Mac or gitbash in Windows) to a directory where you want the repository to be. Then write these commands to clone the project on your pc:

    - `git clone https://github.com/FinbarArgus/circulatory_autogen`

    If you want to develop the code, then create a fork of the above repo in GitHub, then do the following lines instead of the above:

    - `git clone https://github.com/<YourUsername>/circulatory_autogen`

    - `git remote add upstream https://github.com/FinbarArgus/circulatory_autogen`



## Directory Definition

In this tutorial, we define the **project_dir** as the directory where the Github Circulatory Autogen project has been cloned. For example, on our computer, this directory is as below:

`[project_dir]: ~/Documents/git_projects/Circulatory_autogen`

Also, the OpenCOR directory is needed for installing the necessary python libraries, which we defined as the **OpenCOR_dir**, e.g.:

`[OpenCOR_dir]: ~/Desktop/OpenCOR-0-8-1-Linux/`

<!-- If you have Windows but would prefer to use a linux distribution for running CA, you could install a virtual Linux machine. One of these virtual linux machines is VirtualBox Oracle, which can be downloaded from [here](https://www.virtualbox.org/). -->
 <!-- To set up the VirtualBox you would need to download the latest version of Ubuntu using this [link](https://ubuntu.com/download/desktop). -->

!!! info
    If running on the ABI HPC, you can use the installed OpenCOR version at the path: **/hpc/farg967/OpenCOR-0-8-1-Linux/** and Ignore the below installation steps, as the libraries are already installed. See [running on hpc](running-on-hpc.md)

## Python and Libraries Installation

To run openCOR, you need to use the Python version that is shipped with openCOR. 

To install required python packages, navigate to `[OpenCOR_dir]` directory and run the below command.

!!! Note
    === "Linux"
        ```
        ./pip install <packagename>
        ```
    === "Mac"
        ```
        ./pythonshell -m pip install <packagename>
        
        ```
    === "Windows"
        ```
        ./pythonshell.bat -m pip install <packagename>
        
        ```

!!! note
    **Required packages for autogeneration**:
    pandas pyyaml rdflib

    **Recommended (but not required) packages for autogeneration (allows for better error checking)**:
    libcellml

    **Required packages for parameter identification**:
    mpi4py sympy

    **Required packages for mcmc bayesian identification**:
    emcee corner schwimmbad tqdm statsmodels
    
    **Required packages for sensitivity analysis **:
    SALib==1.5.1 seaborn
    
    **Required packages for CMAES**:
    nevergrad
    
    **Required packages for testing**:
    pytest

    **Required for some utilities**:
    ruamel.yaml

## Setting up your python path

Open `[project_dir]/user_run_files/opencor_pythonshell_path.sh` file and change the `opencor_pythonshell_path` to the directory of pythonshell in the **OpenCOR_dir**: 

!!! Note
    === "Linux and Mac"
        ```
        opencor_pythonshell_path=`<OpenCOR_dir>/pythonshell`.
        ```

    === "Windows"
        ```
        opencor_pythonshell_path=`C:\<OpenCOR_dir>\pythonshell.bat`.
        
        Note that the windows path conventions need to be used with C: and "\ rather than "/".
        ```

!!! Note
    This tutorial assumes you will be running .sh commands (if you're on Windows, you should download gitbash from [here](https://git-scm.com/downloads) so that you can run the bash scripts). 

    However, alternatively (**especially if you want to debug**), you can use an IDE of your choice, set the python path equal to the `opencor_pythonshell_path` and run the python scripts that are called to within the bash scripts (open the relevent .sh file to find the corresponding python script name). The scripts are located at `project_dir]/src/scripts/`

!!! warning
    Installing **mpi4py** requires mpi to be available. Therefore, the following lines may be required to install the mpi software on your computer.

    === "Linux"
        ```
        sudo apt install libopenmpi-dev
        sudo apt install libffi7
        ```

    === "Mac"
        ```
        brew install openmpi
        ```
    === "Windows"

        '''
        To be able to import mpi4py, you may have to do the following:

        Download MS MPI, install both .mis and SDK.

        Set up environmental variables. Open `Control Panel` and select `Advanced System Settings`. Then select `Environmental Variables` and add the following.

            C:\Program Files\Microsoft MPI\
            C:\Program Files (x86)\Microsoft SDKs\MPI\
        '''

!!! warning 
    In versions of **OpenCOR < 0.8** you needed to nagivate to the `[OpenCOR_dir]/python/bin` directory and run the below command instead.

    ```
    ./python -m pip install <packagename>
    ```

!!! warning
    For **OpenCOR < 0.8**
    if you get an SSL error you must do the following before the pip install:

        cd [OpenCOR_dir]/python/bin
        export LD_LIBRARY_PATH=[OpenCOR_dir]/lib

    This would let the system know where to look for libcrypto.so.3 when loading the ssl module.


!!! info "Changes to be made"
    Currently **vessels** is used interchangeabley with **modules**. This will be changed to use **modules** in all instances, as the project now allows all types of modules, not just vessels.

    The connections between terminals and the venous system is hardcoded, as a terminal_venous_connection has to be made to sum up the flows and get an avergae concentration. This is being improved in development.
