cmake_minimum_required(VERSION 3.18)
project(smoke_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(smoke_test src/smoke_test.cpp)
target_link_libraries(smoke_test "${TORCH_LIBRARIES}")

add_executable(test_eth_ucy_loader src/eth_ucy_loader.cpp src/test_eth_ucy_loader.cpp)
target_link_libraries(test_eth_ucy_loader "${TORCH_LIBRARIES}")

add_executable(test_metrica_loader src/metrica_loader.cpp src/test_metrica_loader.cpp)
target_link_libraries(test_metrica_loader "${TORCH_LIBRARIES}")


add_executable(test_graph src/graph.cpp src/test_graph.cpp)
target_link_libraries(test_graph "${TORCH_LIBRARIES}")

add_executable(test_integration_real src/eth_ucy_loader.cpp src/graph.cpp src/encoder.cpp
                                       src/domain_adapt.cpp src/heads.cpp src/losses.cpp
                                       src/test_integration_real.cpp)
target_link_libraries(test_integration_real "${TORCH_LIBRARIES}")

add_executable(train src/eth_ucy_loader.cpp src/metrica_loader.cpp src/data_pool.cpp
                       src/graph.cpp src/encoder.cpp src/domain_adapt.cpp src/heads.cpp
                       src/losses.cpp src/train.cpp)
target_link_libraries(train "${TORCH_LIBRARIES}")



add_executable(export_showcase src/graph.cpp src/encoder.cpp src/domain_adapt.cpp src/heads.cpp src/data_pool.cpp src/eth_ucy_loader.cpp src/metrica_loader.cpp src/export_showcase.cpp)
target_link_libraries(export_showcase "${TORCH_LIBRARIES}")
