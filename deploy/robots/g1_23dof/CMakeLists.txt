cmake_minimum_required(VERSION 3.12)
project(g1_29dof_controller)

include_directories(
    include
    ../../include
    /opt/unitree_robotics/include
    /opt/unitree_robotics/include/ddscxx
    /opt/cyclonedds/include
    /opt/cyclonedds/include/ddscxx
)

link_directories(
    /opt/unitree_robotics/lib
    /opt/cyclonedds/lib
)


include_directories(
    include
    ../../include
    /opt/unitree_robotics/include
    /opt/cyclonedds/include
)

link_directories(
    /opt/unitree_robotics/lib
    /opt/cyclonedds/lib
)


include_directories(
    include
    ../../include
    /opt/unitree_robotics/include
)

link_directories(
    /opt/unitree_robotics/lib
)


find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(g1_ctrl main.cpp)

# ---- manually added include/link paths for Unitree SDK2 DDS wrapper ----
target_include_directories(g1_29dof_controller_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    /opt/unitree_robotics/include
    /opt/unitree_robotics/include/ddscxx
)

target_include_directories(g1_ctrl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    /opt/unitree_robotics/include
    /opt/unitree_robotics/include/ddscxx
)

target_link_directories(g1_29dof_controller_lib PUBLIC
    /opt/unitree_robotics/lib
)

target_link_directories(g1_ctrl PRIVATE
    /opt/unitree_robotics/lib
)
# ---- end manually added paths ----
