cmake_minimum_required(VERSION 3.10)
project(MaxQuasiClique)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler warnings
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Add optimization flags for release build
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Include directories
include_directories(include)

# Add executables
add_executable(max_quasi_clique src/main.cpp)
add_executable(optimized_solver src/optimized_solver.cpp)

# Add threading support
find_package(Threads REQUIRED)
target_link_libraries(max_quasi_clique PRIVATE Threads::Threads)
target_link_libraries(optimized_solver PRIVATE Threads::Threads)

# Copy data files to build directory
file(COPY data DESTINATION ${CMAKE_BINARY_DIR})