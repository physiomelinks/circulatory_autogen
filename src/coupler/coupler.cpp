#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <cstdio>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <poll.h>
#include <map>
// #include <omp.h> //to use OpenMP API for parallel programming
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "coupler_auxFun.h"

class ProcessManager {
private:
    pid_t process_one = -1;
    pid_t process_zero = -1;

    std::string python_path;
    const char* char_python_path;
    std::string path1d;
    const char* char_path1d;
    std::string path0d;
    const char* char_path0d;
    std::string initFile1d;
    const char* char_initFile1d;

    std::string inputFold;
    std::string networkName;
    const char* char_networkName;
    std::string ODEsolver;
    const char* char_ODEsolver;
    double T0;
    const char* char_T0;
    int nCC;
    const char* char_nCC;

    std::string initStatePath;
    const char* char_initStatePath;

    size_t N1d0dTot;
    size_t N1d0d;
    AuxCouplerFun* auxFun;
    bool couple_volume_sum = false;

    const double tolTime = 1e-11; // 1e-13;
    const size_t N = 1024;
  
    // Named pipe paths
    std::vector<const char*> one_to_parent;
    std::vector<const char*> parent_to_one;
    std::vector<const char*> zero_to_parent;
    std::vector<const char*> parent_to_zero;

    std::string user_pipePath;
    const char* char_user_pipePath;

    std::string pipePath_dt1, pipePath_dt2, pipePath_dt3, pipePath_dt4;
    std::string pipePath_vol1, pipePath_vol2;

    const char* one_to_parent_dt;
    const char* parent_to_one_dt;
    const char* zero_to_parent_dt;
    const char* parent_to_zero_dt;

    const char* one_to_parent_vol;
    const char* parent_to_zero_vol;


    void cleanup() {
        if (process_one > 0) kill(process_one, SIGTERM);
        if (process_zero > 0) kill(process_zero, SIGTERM);
        
        // Remove named pipes
        for (size_t i = 0; i < N1d0d; ++i) {
            unlink(one_to_parent[i]);
            unlink(parent_to_one[i]);
            unlink(zero_to_parent[i]);
            unlink(parent_to_zero[i]);
        }

        unlink(one_to_parent_dt);
        unlink(parent_to_one_dt);
        unlink(zero_to_parent_dt);
        unlink(parent_to_zero_dt);

        unlink(one_to_parent_vol);
        unlink(parent_to_zero_vol);

        // Free each dynamically allocated string
        for (auto& pipe : one_to_parent) {
            free((void*)pipe);
            pipe = nullptr;
        }
        for (auto& pipe : parent_to_one) {
            free((void*)pipe);
            pipe = nullptr;
        }
        for (auto& pipe : zero_to_parent) {
            free((void*)pipe);
            pipe = nullptr;
        }
        for (auto& pipe : parent_to_zero) {
            free((void*)pipe);
            pipe = nullptr;
        }
    }

public:
    ProcessManager() : auxFun(nullptr) {}

        void init(const std::string& initFilePath) {

            std::ifstream file(initFilePath);
            json config;
            file >> config;

            inputFold = config.value("inputFold", "./../CA_files/generated_models/aortic_bif_hybrid/");
            networkName = config.value("networkName", "aortic_bif_hybrid");
            ODEsolver = config.value("ODEsolver", "CVODE");
            T0 = config.value("T0", 1.1);
            nCC = config.value("nCC", 5);
            
            user_pipePath = config.value("tmp_pipe_path", "/home/bghi639/Software/tmp/");
            char_user_pipePath = user_pipePath.c_str();

            initStatePath = config.value("initStatePath", "None");
            char_initStatePath = initStatePath.c_str();

            // path to python executable
            python_path = config.value("python_path", "/home/bghi639/anaconda3/bin/python"); // or "/hpc/bghi639/anaconda3/bin/python" or "/usr/bin/python3"
            char_python_path = python_path.c_str();
            // path to python script
            path1d = config.value("solver1d_path", "./solver1D/main1D.py"); 
            char_path1d = path1d.c_str();
            // path to cpp executable
            path0d = config.value("solver0d_path", "./solver0D/main0d");
            char_path0d = path0d.c_str();
            // path to 1d simulation initialisation file
            initFile1d = config.value("initFile_sim1d_path", "./../Python1D_files/aortic_bif_hybrid/run000/input.ini");
            char_initFile1d = initFile1d.c_str();
   
            char_networkName = networkName.c_str();
            char_ODEsolver = ODEsolver.c_str();

            std::ostringstream oss1;
            oss1 << std::fixed << std::setprecision(8) << T0; // 8 decimal places should be enough for T0
            std::string str1 = oss1.str(); 
            char_T0 = str1.c_str();
            
            std::string str2 = std::to_string(nCC); // integer
            char_nCC = str2.c_str();
              
            auxFun = new AuxCouplerFun();

            std::string fileJson = inputFold+networkName+"_coupler1d0d.json";
            std::string dataJson = auxFun->readJsonFromFile(fileJson);
            
            std::map<std::string, std::map<std::string, int>> pipe_1d_0d_info = auxFun->deserializeFromJson(dataJson);
            N1d0dTot = pipe_1d_0d_info.size();
            N1d0d = 0;
            // XXX here we are assuming that these 1d-0d connections are ordered such that the volume sum connection is the last one.
            // TODO make sure this is always the case, by ordering the connections based on their type.
            for (size_t i = 0; i < N1d0dTot; ++i) {
                if (pipe_1d_0d_info[std::to_string(i+1)].find("port_volume_sum") != pipe_1d_0d_info[std::to_string(i+1)].end()){
                    if (pipe_1d_0d_info[std::to_string(i+1)]["port_volume_sum"] == 1){
                        // this is a 'global' volume sum connection, for which we will define dedicated pipes (as for the time step)
                        couple_volume_sum = true;
                    }
                // here we could add other 'special' global connections for which we want dedicated pipes
                // we are assuming that all these special connections are stored at the end of the pipe_1d_0d_info file/structure
                } else{
                    // this is a 'standard' one-to-one 1d-0d connection
                    N1d0d++;
                }
            }
            std::cout << "Number of one-to-one 1d-0d connections: " << N1d0d << std::endl;
            std::cout << "Total number of connections (including global connections): " << N1d0dTot << " || With time step connection: "  << N1d0dTot+1 << std::endl;

            pipePath_dt1 = user_pipePath+std::string("one_to_parent_dt");
            one_to_parent_dt = pipePath_dt1.c_str();
            pipePath_dt2 = user_pipePath+std::string("parent_to_one_dt");
            parent_to_one_dt = pipePath_dt2.c_str();
            pipePath_dt3 = user_pipePath+std::string("zero_to_parent_dt");
            zero_to_parent_dt = pipePath_dt3.c_str();
            pipePath_dt4 = user_pipePath+std::string("parent_to_zero_dt");
            parent_to_zero_dt = pipePath_dt4.c_str();

            if (couple_volume_sum){
                pipePath_vol1 = user_pipePath+std::string("one_to_parent_vol");
                one_to_parent_vol = pipePath_vol1.c_str();
                pipePath_vol2 = user_pipePath+std::string("parent_to_zero_vol");
                parent_to_zero_vol = pipePath_vol2.c_str();
            }


            for (size_t i = 0; i < N1d0d; ++i) {
                int pipeID = i+1;
                std::string pipePath;

                pipePath = user_pipePath+std::string("one_to_parent_")+std::to_string(pipeID);
                one_to_parent.push_back(strdup(pipePath.c_str()));

                pipePath = user_pipePath+std::string("parent_to_one_")+std::to_string(pipeID);
                parent_to_one.push_back(strdup(pipePath.c_str()));

                pipePath = user_pipePath+std::string("zero_to_parent_")+std::to_string(pipeID);
                zero_to_parent.push_back(strdup(pipePath.c_str()));

                pipePath = user_pipePath+std::string("parent_to_zero_")+std::to_string(pipeID);
                parent_to_zero.push_back(strdup(pipePath.c_str()));            
            }

            // Create Named Pipes (FIFOs), instead of "standard" pipes
            if ((mkfifo(one_to_parent_dt, 0666) < 0 && errno != EEXIST) ||
                (mkfifo(parent_to_one_dt, 0666) < 0 && errno != EEXIST) ||
                (mkfifo(zero_to_parent_dt, 0666) < 0 && errno != EEXIST) ||
                (mkfifo(parent_to_zero_dt, 0666) < 0 && errno != EEXIST)) {
                std::cerr << "mkfifo failed: " << strerror(errno) << std::endl;
                throw std::runtime_error("Failed to create named pipes for time step");
            }

            for (size_t i = 0; i < N1d0d; ++i) {
                if ((mkfifo(one_to_parent[i], 0666) < 0 && errno != EEXIST) ||
                    (mkfifo(parent_to_one[i], 0666) < 0 && errno != EEXIST) ||
                    (mkfifo(zero_to_parent[i], 0666) < 0 && errno != EEXIST) ||
                    (mkfifo(parent_to_zero[i], 0666) < 0 && errno != EEXIST)) {
                    std::cerr << "mkfifo failed: " << strerror(errno) << std::endl;
                    throw std::runtime_error("Failed to create named pipes for 1d-0d connections");
                }
            }

            if (couple_volume_sum){
                if ((mkfifo(one_to_parent_vol, 0666) < 0 && errno != EEXIST) ||
                    (mkfifo(parent_to_zero_vol, 0666) < 0 && errno != EEXIST)) {
                    std::cerr << "mkfifo failed: " << strerror(errno) << std::endl;
                    throw std::runtime_error("Failed to create named pipes for time step");
                }
            }

        std::cout << "Coupler :: Initialization completed successfully." << std::endl;

        }

        void start_processes() {
            // Start 1D solver main1D.py
            process_one = fork();
            if (process_one == 0) { // Child process for main1D.py
                execl(char_python_path, 
                    "python", 
                    char_path1d, 
                    "1", 
                    char_ODEsolver, 
                    char_T0, 
                    char_nCC, 
                    char_initFile1d, 
                    char_user_pipePath,
                    char_initStatePath,
                    (char*)nullptr); // nullptr); 

                perror("process_one :: execl failed");
                exit(1);
            }

            // Start 0D solver main0d.cpp
            process_zero = fork();
            if (process_zero == 0) { // Child process for main0d.cpp
                // execl(char_path0d, 
                //     "main0d", 
                //     "1", 
                //     char_ODEsolver, 
                //     char_T0, 
                //     char_nCC, 
                //     char_networkName, 
                //     char_user_pipePath, 
                //     (char*)nullptr); // nullptr);    
                // perror("process_zero :: execl failed");
                // exit(1);

                std::vector<std::string> args_vec;
                std::vector<char*> args_ptr;
                bool use_mpirun = false;
                if (ODEsolver == "PETSC"){
                    use_mpirun = true;
                }
                
                if (use_mpirun) {
                    args_vec.insert(args_vec.end(), { "mpirun", "-np", "1", char_path0d });
                } else {
                    args_vec.push_back(char_path0d);
                }
                
                // XXX for now, we are manually setting the time stepping method for PETSc in model0d.cpp
                // TODO make this more flexible by adding an input argument to main0d and passing it here 
                // if (use_mpirun) {
                //     // args_vec.insert(args_vec.end(), { "-ts_type", "bdf", "-ts_bdf_order", "2" }); // BDF2
                //     args_vec.insert(args_vec.end(), { "-ts_type", "beuler" }); // Backward Euler
                //     // args_vec.insert(args_vec.end(), { "-ts_type", "cn" }); // Crank-Nicolson
                // }

                // if (use_mpirun) {
                //     // args_vec.insert(args_vec.end(), { "-ts_type", "bdf", "-ts_bdf_order", "2" }); // BDF2
                //     // args_vec.insert(args_vec.end(), { "-ts_max_snes_failures", "-1" }); // Backward Euler
                //     args_vec.insert(args_vec.end(), { "-snes_rtol", "1e-4" });
                //     // args_vec.insert(args_vec.end(), { "-ts_type", "cn" }); // Crank-Nicolson
                // }

                args_vec.insert(args_vec.end(),
                                { "-coupled", "1", 
                                "-ODEsolver", char_ODEsolver, 
                                "-T0", char_T0,
                                "-nCC", char_nCC,
                                "-networkName", char_networkName,
                                "-pipePath", char_user_pipePath,
                                "-initStatePath", char_initStatePath });

                for (auto& s : args_vec) {
                    args_ptr.push_back(const_cast<char*>(s.c_str()));
                }
                args_ptr.push_back(nullptr);

                execvp(args_ptr[0], args_ptr.data());
                perror("process_zero :: execl failed");
                exit(1);

            }

            // Parent process handles communication between processes
            // The three processes (zero, one & coupler) are running "in parallel" 
            std::thread relay_thread([this]() {

                double dtGlob = 0.;
                double timeGlob = 0.;
                double dtLoc = 0.;
                double timeLoc = 0.;
                // double tEndGlob = nCC*T0;
                
                std::vector<std::vector<char>> one_buffer(N1d0d, std::vector<char>(N));
                std::vector<std::vector<char>> zero_buffer(N1d0d, std::vector<char>(N));

                std::vector<char> one_buffer_dt(N);
                std::vector<char> zero_buffer_dt(N);
                std::vector<char> one_buffer_vol(N);

                int count = 0;
                
                // Open named pipes
                int one_read_fd[N1d0d]; 
                int one_write_fd[N1d0d];
                int zero_read_fd[N1d0d];
                int zero_write_fd[N1d0d];

                int one_read_dt_fd = open(one_to_parent_dt, O_RDONLY);
                int zero_read_dt_fd = open(zero_to_parent_dt, O_RDONLY);
                for (size_t i = 0; i < N1d0d; ++i) {
                    one_read_fd[i] = open(one_to_parent[i], O_RDONLY);
                    zero_read_fd[i] = open(zero_to_parent[i], O_RDONLY);
                }
                int one_write_dt_fd = open(parent_to_one_dt, O_WRONLY);
                int zero_write_dt_fd = open(parent_to_zero_dt, O_WRONLY);
                for (size_t i = 0; i < N1d0d; ++i) {
                    one_write_fd[i] = open(parent_to_one[i], O_WRONLY);
                    zero_write_fd[i] = open(parent_to_zero[i], O_WRONLY);
                }

                int one_read_vol_fd = -1;
                int zero_write_vol_fd = -1;
                if (couple_volume_sum){
                    one_read_vol_fd = open(one_to_parent_vol, O_RDONLY);
                    zero_write_vol_fd = open(parent_to_zero_vol, O_WRONLY);
                }

                bool time_step_exchanged = true;
                bool all_pipes_closed = true;
                
                while (true) { 

                    all_pipes_closed = true;
                    // this is to initialise volume sum exchange from main1d to main0d, to compute total volume in the system
                    if (count==0 && couple_volume_sum){
                        if (one_read_vol_fd!=-1 && zero_write_vol_fd!=-1){
                            // read volume sum data from main1D
                            ssize_t N_one = read(one_read_vol_fd, one_buffer_vol.data(), one_buffer_vol.size());
                            if (N_one <= 0) {
                                std::cerr << "Coupler :: Error :: Volume sum read from 1d failed or returned 0 bytes" << std::endl;
                            }
                            else{
                                all_pipes_closed = false;
                            }
                            size_t N_one_wr = static_cast<size_t>(N_one);
                        
                            N_one_wr = std::min(N_one_wr, one_buffer_vol.size());
                            // write and send volume sum data to main0d
                            ssize_t N_zero = write(zero_write_vol_fd, one_buffer_vol.data(), N_one_wr);
                            if (N_zero < 0) {
                                std::cerr << "Coupler :: Error :: Volume sum write to 0d failed" << std::endl;
                            }
                        }
                    }

                    // this is to exchange global synchronization time and time step between main0d and main1D BEFORE advancing each solver in time
                    if (time_step_exchanged){
                        // read time step data from main0d
                        ssize_t N_zero = read(zero_read_dt_fd, zero_buffer_dt.data(), zero_buffer_dt.size());
                        if (N_zero <= 0) {
                            std::cerr << "Coupler :: Error :: Time step read from 0d failed or returned 0 bytes" << std::endl;
                        } else{
                            all_pipes_closed = false;
                        }
                        size_t N_zero_wr = static_cast<size_t>(N_zero);
                        
                        N_zero_wr = std::min(N_zero_wr, zero_buffer_dt.size());
                        // write and send time step data to main1D
                        ssize_t N_one = write(one_write_dt_fd, zero_buffer_dt.data(), N_zero_wr);
                        if (N_one < 0) {
                            std::cerr << "Coupler :: Error :: Time step write to 1d failed" << std::endl;
                        }
                        
                        // read time step data from main1D
                        N_one = read(one_read_dt_fd, one_buffer_dt.data(), one_buffer_dt.size());
                        if (N_one <= 0) {
                            std::cerr << "Coupler :: Error :: Time step read from 1d failed or returned 0 bytes" << std::endl;
                        }
                        else{
                            all_pipes_closed = false;
                        }
                        size_t N_one_wr = static_cast<size_t>(N_one);
                        
                        N_one_wr = std::min(N_one_wr, one_buffer_dt.size());
                        // write and send time step data back to main0d
                        N_zero = write(zero_write_dt_fd, one_buffer_dt.data(), N_one_wr);
                        if (N_zero < 0) {
                            std::cerr << "Coupler :: Error :: Time step write to 0d failed" << std::endl;
                        }
                        
                        if (all_pipes_closed){
                            std::cout << "Coupler :: All pipes closed, relay thread exiting..." << std::endl;
                            break;
                        }

                        // convert time step data from bytes to doubles
                        std::vector<double> data_one_dt_read = auxFun->extractDoublesFromBytes(one_buffer_dt.data(), one_buffer_dt.size(), N_one_wr);
                        timeGlob = data_one_dt_read[0];
                        dtGlob = data_one_dt_read[1];
                        std::cout << std::setprecision(8) << "Coupler :: Global time exchanged : " << timeGlob << " || dt : " << dtGlob << std::endl;
                    }

                    // this is to exchange local internal time and time step taken by the specific ODE solver between main0d and main1D
                    // read time step data from main0d
                    ssize_t N_zero = read(zero_read_dt_fd, zero_buffer_dt.data(), zero_buffer_dt.size());
                    if (N_zero <= 0) {
                        std::cerr << "Coupler :: Error :: Time step read from 0d failed or returned 0 bytes" << std::endl;
                    } else{
                        all_pipes_closed = false;
                    }
                    size_t N_zero_wr = static_cast<size_t>(N_zero);

                    N_zero_wr = std::min(N_zero_wr, zero_buffer_dt.size());
                    std::vector<double> data_zero_dt_read = auxFun->extractDoublesFromBytes(zero_buffer_dt.data(), zero_buffer_dt.size(), N_zero_wr);
                    timeLoc = data_zero_dt_read[0];
                    dtLoc = data_zero_dt_read[1];

                    // std::cout << "Coupler :: Local time step exchanged " << timeLoc << " " << dtLoc << std::endl;

                    time_step_exchanged = false;
                    if (dtLoc<0.){ // XXX dtLoc = -999 indicates that the ODE solver has completed the evolution to the next time level
                        // std::cout << "Coupler :: Received negative 0d internal time step dtLoc " << dtLoc << std::endl;

                        // write and send time step data to main1D
                        ssize_t N_one = write(one_write_dt_fd, zero_buffer_dt.data(), N_zero_wr);
                        if (N_one < 0) {
                            std::cerr << "Coupler :: Error :: Time step write to 1d failed" << std::endl;
                        }
                        
                        N_one = read(one_read_dt_fd, one_buffer_dt.data(), one_buffer_dt.size());
                        if (N_one <= 0) {
                            std::cerr << "Coupler :: Error :: Time step read from 1d failed or returned 0 bytes" << std::endl;
                        }
                        else{
                            all_pipes_closed = false;
                        }
                        size_t N_one_wr = static_cast<size_t>(N_one);
                        
                        N_one_wr = std::min(N_one_wr, one_buffer_dt.size());
                        // write and send time step data back to main0d
                        N_zero = write(zero_write_dt_fd, one_buffer_dt.data(), N_one_wr);
                        if (N_zero < 0) {
                            std::cerr << "Coupler :: Error :: Time step write to 0d failed" << std::endl;
                        }

                        if (couple_volume_sum){
                            if (one_read_vol_fd!=-1 && zero_write_vol_fd!=-1){
                                // read volume sum data from main1D
                                ssize_t N_one = read(one_read_vol_fd, one_buffer_vol.data(), one_buffer_vol.size());
                                if (N_one <= 0) {
                                    std::cerr << "Coupler :: Error :: Volume sum read from 1d failed or returned 0 bytes" << std::endl;
                                }
                                else{
                                    all_pipes_closed = false;
                                }
                                size_t N_one_wr = static_cast<size_t>(N_one);

                                N_one_wr = std::min(N_one_wr, one_buffer_vol.size());
                                // write and send volume sum data to main0d
                                ssize_t N_zero = write(zero_write_vol_fd, one_buffer_vol.data(), N_one_wr);
                                if (N_zero < 0) {
                                    std::cerr << "Coupler :: Error :: Volume sum write to 0d failed" << std::endl;
                                }
                            }
                        }
                        
                        if (all_pipes_closed){
                            std::cout << "Coupler :: All pipes closed, relay thread exiting..." << std::endl;
                            break;
                        }
                        
                        count++;
                        time_step_exchanged = true;
                        continue;
                    }

                    // if (timeLoc>timeGlob+dtGlob){
                    if (std::round(timeLoc/tolTime)*tolTime > std::round((timeGlob+dtGlob)/tolTime)*tolTime) {
                        // but this should not happen
                        dtLoc = dtLoc - (timeLoc-(timeGlob+dtGlob));
                        timeLoc = timeGlob+dtGlob;
                        timeLoc = std::round(timeLoc/tolTime)*tolTime;
                        // std::cout << "Coupler :: LOCAL TIME LARGER THAN GLOBAL TIME " << timeLoc << " " << timeGlob+dtGlob << " || NEW TIME STEP " << " " << dtLoc << std::endl;
                        
                        data_zero_dt_read[0] = timeLoc;
                        data_zero_dt_read[1] = dtLoc;
                        auxFun->convertDoublesToBytes(data_zero_dt_read, zero_buffer_dt.data(), N_zero_wr);
                    }
                    
                    // write and send time step data to main1D
                    ssize_t N_one = write(one_write_dt_fd, zero_buffer_dt.data(), N_zero_wr);
                    if (N_one < 0) {
                        std::cerr << "Coupler :: Error :: Time step write to 1d failed" << std::endl;
                    }

                    // std::cout << std::setprecision(16)
                    //         << "Coupler :: timeGlobal " <<  timeGlob << " " << dtGlob 
                    //         << " || timeLocal " <<  timeLoc << " " << dtLoc 
                    //         << " || timeDiff " << (timeGlob+dtGlob)-timeLoc
                    //         << std::endl; 

                    // this is to exchange solution data between main0d and main1D for each 1D-0D connection
                    for (size_t i = 0; i < N1d0d; ++i) {
                        // read data from main0d for each 1D-0D connection
                        ssize_t N_zero = read(zero_read_fd[i], zero_buffer[i].data(), zero_buffer[i].size());
                        if (N_zero <= 0) {
                            std::cerr << "Coupler :: Error for pipe " << i << " :: Data read from 0d failed or returned 0 bytes" << std::endl;
                            continue;  // skip this pipe but continue with others
                        }
                        all_pipes_closed = false;
                        size_t N_zero_wr = static_cast<size_t>(N_zero);
                        
                        N_zero_wr = std::min(N_zero_wr, zero_buffer[i].size());
                        // write and send data to main1D for each 1D-0D connection
                        ssize_t N_one = write(one_write_fd[i], zero_buffer[i].data(), N_zero_wr);
                        if (N_one < 0) {
                            std::cerr << "Coupler :: Error for pipe " << i << " :: Data write to 1d failed" << std::endl;
                        }
                    }

                    for (size_t i = 0; i < N1d0d; ++i) {
                        // read data from main1D for each 1D-0D connection
                        ssize_t N_one = read(one_read_fd[i], one_buffer[i].data(), one_buffer[i].size());
                        if (N_one <= 0) {
                            std::cerr << "Coupler :: Error for pipe " << i << " :: Data read from 1d failed or returned 0 bytes" << std::endl;
                            continue;  // skip this pipe but continue with others
                        }
                        all_pipes_closed = false;
                        size_t N_one_wr = static_cast<size_t>(N_one);
                        
                        N_one_wr = std::min(N_one_wr, one_buffer[i].size());
                        // write and send data back to main0d for each 1D-0D connection
                        ssize_t N_zero = write(zero_write_fd[i], one_buffer[i].data(), N_one_wr);
                        if (N_zero < 0) {
                            std::cerr << "Coupler :: Error for pipe " << i << " :: Data write to 0d failed" << std::endl;
                        }
                    }

                    count++;
                    if (all_pipes_closed) {
                        std::cout << "All pipes closed, relay thread exiting..." << std::endl;
                        break;
                    }
                }

                std::cout << "### Coupler :: Stop execution! ###" << std::endl;
                std::cout << "Global final time : " << std::round((timeGlob+dtGlob)/tolTime)*tolTime << std::endl;
                std::cout << "Number of 1D-0D communications : " << count << std::endl;

                // Close named pipes
                for (size_t i = 0; i < N1d0d; ++i) {
                    close(one_read_fd[i]);
                    close(one_write_fd[i]);
                    close(zero_read_fd[i]);
                    close(zero_write_fd[i]);
                }
                close(one_read_dt_fd);
                close(one_write_dt_fd);
                close(zero_read_dt_fd);
                close(zero_write_dt_fd);
                
                if (couple_volume_sum){
                    close(one_read_vol_fd);
                    close(zero_write_vol_fd);
                }
            });

            // Wait for both processes
            int one_status, zero_status;
            waitpid(process_one, &one_status, 0);
            waitpid(process_zero, &zero_status, 0);

            std::cout << "Coupler :: Waiting for relay thread to join..." << std::endl;
            relay_thread.join();
            
            std::cout << "### Coupler :: Both processes terminated successfully with status codes : " << one_status << " " << zero_status << " ###" << std::endl;
            
            cleanup();

            std::cout << "### DONE ###" << std::endl;

        }

        ~ProcessManager() {
            cleanup();
        }
};

int main(int argc, char* argv[]) {
    // omp_set_num_threads(4);

    try {
        std::string init_file_path;
        if (argc > 1) {
            init_file_path = argv[1];
        } else {
            std::cerr << "Coupler :: Error: Input file path needs to be passed as command-line argument when executing your coupler. Exiting." << std::endl;
            return 1;
        }

        ProcessManager manager;
        manager.init(init_file_path);
        manager.start_processes();
    } 
    catch (const std::exception& e) {
        std::cerr << "Coupler :: Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
