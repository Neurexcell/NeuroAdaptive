cmake_minimum_required(VERSION 3.10)
project(NeuroAdaptiveAI)

set(CMAKE_CXX_STANDARD 17)

# Set up the include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    src/main.cpp
    src/neural_network.cpp
    src/data_handler.cpp
)

# Create executable
add_executable(NeuroAdaptiveAI ${SOURCES})

# Add these lines to include the new source files
add_executable(NeuroAdaptiveAI main.cpp
    neural_network.cpp
    activation_functions.cpp
    dropout.cpp
    serialization.cpp
)
