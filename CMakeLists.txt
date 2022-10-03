cmake_minimum_required(VERSION 3.6)

project(culll
VERSION 0.1.0
DESCRIPTION "LLL algorithm implemented with CUDA for GPU acceleration"
LANGUAGES CXX)



set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)


include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./extern/pybind11/include
    )

include_directories(include)

link_directories(
    /usr/local/cuda-11.4/lib64
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#file(GLOB_RECURSE SOURCES src/**.cpp)
file(GLOB CU_SOURCES src/**.cu)

cuda_add_library(culll SHARED ${CU_SOURCES} ${SOURCES})

target_link_libraries(culll
  ${PYTHON_LIBRARIES}
  cudart)


# print python
message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")

set_target_properties(culll PROPERTIES PREFIX "")