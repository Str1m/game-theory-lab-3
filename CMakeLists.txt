cmake_minimum_required(VERSION 3.21)
project(lab3)

set(CMAKE_CXX_STANDARD 20)

find_package(Armadillo REQUIRED)

message(STATUS "Armadillo library status:")
message(STATUS "    config: ${ARMADILLO_CONFIG}")
message(STATUS "    include path: ${ARMADILLO_INCLUDE_DIR}")
message(STATUS "    libraries: ${ARMADILLO_LIBRARIES}")

include_directories(${ARMADILLO_INCLUDE_DIRS})
add_executable(lab3 main.cpp)
target_link_libraries(lab3 ${ARMADILLO_LIBRARIES})