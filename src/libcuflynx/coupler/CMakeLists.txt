# The 1D-0D coupler: launches the generated 0D model (main0d) and the Python FV 1D solver
# (libcuflynx/solver1d/main1D.py) and relays their messages over named pipes.
#
#   cmake -S . -B build && cmake --build build
#
# nlohmann/json (header only) is used from the system if installed, otherwise downloaded at
# configure time. Set -DCOUPLER_FETCH_JSON=OFF to require a system install.
cmake_minimum_required(VERSION 3.16)
project(cuflynx_coupler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(COUPLER_FETCH_JSON "Download nlohmann/json if it is not installed" ON)

find_package(nlohmann_json 3 QUIET)
if(NOT nlohmann_json_FOUND)
    if(NOT COUPLER_FETCH_JSON)
        message(FATAL_ERROR "nlohmann_json not found (install nlohmann-json3-dev, or configure with -DCOUPLER_FETCH_JSON=ON)")
    endif()
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
    FetchContent_MakeAvailable(nlohmann_json)
    message(STATUS "Using downloaded nlohmann_json 3.11.3")
endif()

find_package(Threads REQUIRED)

add_executable(coupler coupler.cpp coupler_auxFun.cpp)
target_link_libraries(coupler PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
