cmake_minimum_required(VERSION 3.16)
project(scan_and_plan_bt_executor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(behaviortree_ros2 REQUIRED)
# find_package(eut_collab_bt_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

set(THIS_PACKAGE_DEPS
    behaviortree_ros2
    std_msgs
    std_srvs
    # btcpp_ros2_interfaces
    )

######################################################
# Simple example showing how to use and customize the BtExecutionServer
add_executable(scan_and_plan_bt_executor src/scan_and_plan_bt_executor.cpp)
ament_target_dependencies(scan_and_plan_bt_executor ${THIS_PACKAGE_DEPS})

######################################################

install(TARGETS
  scan_and_plan_bt_executor
  DESTINATION lib/${PROJECT_NAME}
  )

######################################################
# INSTALL Behavior.xml's, ROS config and launch files

install(DIRECTORY
    behavior_trees
    config
    launch
    DESTINATION share/${PROJECT_NAME}/
    )

    ament_export_dependencies(behaviortree_ros2)

# ament_export_dependencies(behaviortree_ros2 eut_collab_bt_interfaces)

ament_package()
