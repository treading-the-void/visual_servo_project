cmake_minimum_required(VERSION 3.0.2)
project(image_decompressor)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  image_transport
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
)

set(CMAKE_PREFIX_PATH "/home/visp_ws/visp-build")
set(VISP_DIR "/home/visp_ws/visp-build" CACHE PATH "VISP CMake directory")

find_package(ur_rtde REQUIRED PATHS "/home/ymy/visp-ws/3rdparty/ur_rtde/build/ur_rtde")
find_package(VISP REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

# 包含目录
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${VISP_INCLUDE_DIRS}
)

# 安装目标
catkin_package()

# 添加可执行文件
add_executable(eyeinhand src/eyeinhand.cpp)
add_executable(trajpre src/trajpre.cpp)

# 链接库
target_link_libraries(eyeinhand
  ${catkin_LIBRARIES}
  ${ur_rtde_LIBRARIES}
  ${VISP_LIBRARIES}
  ${Boost_LIBRARIES}
)
target_link_libraries(trajpre
  ${catkin_LIBRARIES}
  ${ur_rtde_LIBRARIES}
  ${VISP_LIBRARIES}   
  ${Boost_LIBRARIES}
)

