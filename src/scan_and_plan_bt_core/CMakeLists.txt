cmake_minimum_required(VERSION 3.5)
project(scan_and_plan_bt_core CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(behaviortree_ros2 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(ur_dashboard_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(scan_and_plan_interfaces REQUIRED)



include_directories(include)

set(dependencies
  rclcpp
  behaviortree_cpp
  behaviortree_ros2
  geometry_msgs
  sensor_msgs
  shape_msgs
  scan_and_plan_interfaces
)

# Create a single shared library for all plugins.
# Add all your plugin source files and the centralized registration file.
add_library(scan_and_plan_bt_plugins SHARED
            src/plugins/action/store_latest_pcd.cpp
            src/plugins/action/filter_pcd.cpp
            src/plugins/action/compute_ellipsoid_fitting.cpp
            src/plugins/action/get_viewpoints.cpp
            src/register_plugins.cpp
)

ament_target_dependencies(scan_and_plan_bt_plugins ${dependencies})
target_compile_definitions(scan_and_plan_bt_plugins PRIVATE BT_PLUGIN_EXPORT)

######################################################
# INSTALL

# Recover subfolder path for plugins
execute_process(
    COMMAND bash -c "xmllint --xpath 'string(//description/plugins/@folder_path)' plugins_description.xml | awk '{print \$1\}'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE PLUGINS_FOLDER_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


# Install the shared library into the desired directory.
install(TARGETS scan_and_plan_bt_plugins
  LIBRARY DESTINATION share/${PROJECT_NAME}/${PLUGINS_FOLDER_PATH}
  ARCHIVE DESTINATION share/${PROJECT_NAME}/${PLUGINS_FOLDER_PATH}
  RUNTIME DESTINATION share/${PROJECT_NAME}/${PLUGINS_FOLDER_PATH}
)

# Commented out: bt_ros_plugins_to_xml not available in this version of behaviortree_ros2
# Plugins are registered manually in register_plugins.cpp instead
#
# # Assume behaviortree_eut_pluginsConfig.cmake is installed into <install_prefix>/share/behaviortree_eut_plugins.
# # Get the install prefix from the config file location:
# get_filename_component(_bt_ros2_share_intern_dir "${behaviortree_ros2_DIR}" DIRECTORY)
# get_filename_component(_bt_ros2_share_dir "${_bt_ros2_share_intern_dir}" DIRECTORY)
# get_filename_component(_bt_ros2_install_dir "${_bt_ros2_share_dir}" DIRECTORY)
#
# # Construct the expected path to the bt_plugins_to_xml executable:
# set(BT_PLUGINS_TO_XML_EXECUTABLE "${_bt_ros2_install_dir}/lib/behaviortree_ros2/bt_ros_plugins_to_xml")
#
# if(NOT EXISTS "${BT_PLUGINS_TO_XML_EXECUTABLE}")
#   message(FATAL_ERROR "bt_ros_plugins_to_xml executable not found. Ensure behaviortree_ros2 is built and installed.")
# endif()
#
# add_custom_command(
#   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libscan_and_plan_bt_plugins.xml
#   COMMAND ${BT_PLUGINS_TO_XML_EXECUTABLE}
#           $<TARGET_FILE:scan_and_plan_bt_plugins>
#           ${CMAKE_CURRENT_BINARY_DIR}/libscan_and_plan_bt_plugins.xml
#   DEPENDS scan_and_plan_bt_plugins
#   COMMENT "Generating XML file from plugin .so"
# )
#
# add_custom_target(generate_xml ALL
#   DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libscan_and_plan_bt_plugins.xml
# )

# Install header files.
install(DIRECTORY include/ DESTINATION include/)

# install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libscan_and_plan_bt_plugins.xml
#         DESTINATION share/${PROJECT_NAME}/${PLUGINS_FOLDER_PATH})


# Register the plugin description file as a resource
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plugins_description.xml
               ${CMAKE_CURRENT_BINARY_DIR}/plugins_description.xml COPYONLY)

ament_index_register_resource(
  "behavior_tree_plugin"
  CONTENT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/plugins_description.xml"
)


ament_export_include_directories(include)
ament_export_libraries(scan_and_plan_bt_plugins)
ament_export_dependencies(${dependencies})

ament_package()
