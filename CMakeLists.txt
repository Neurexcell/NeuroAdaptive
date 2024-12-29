cmake_minimum_required(VERSION 3.10)
project(NeuroAdaptiveAI)

set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/neural_network.cpp
    src/data_handler.cpp
    add_subdirectory(src/training)
)

add_executable(NeuroAdaptiveAI ${SOURCES})

add_executable(NeuroAdaptiveAI main.cpp
    neural_network.cpp
    activation_functions.cpp
    dropout.cpp
    serialization.cpp
)
