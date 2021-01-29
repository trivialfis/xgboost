cmake_minimum_required(VERSION 3.13)
project(external-memory-demo LANGUAGES C CXX VERSION 0.0.1)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../cmake/modules")
find_package(OpenMP REQUIRED)
find_package(xgboost REQUIRED)

add_executable(external-memory-demo external_memory.c)
target_link_libraries(external-memory-demo PRIVATE xgboost::xgboost OpenMP::OpenMP_CXX)
