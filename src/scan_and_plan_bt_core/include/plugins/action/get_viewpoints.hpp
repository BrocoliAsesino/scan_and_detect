#pragma once

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "scan_and_plan_interfaces/srv/generate_view_points.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "shape_msgs/msg/mesh.hpp"
#include <behaviortree_ros2/bt_service_node.hpp>

#include <string>
#include <vector>

/**
 * @brief Behavior Tree Service Node for camera viewpoint generation around
 * objects.
 *
 * This node calls the GenerateViewPoints service to compute optimal camera
 * positions for robotic scanning based on ellipsoid approximation.
 *
 * Viewpoints can be generated:
 * - Around the ellipsoid at specified elevation angles
 * - Along principal axis (major or minor) looking down
 *
 * Input Ports:
 * - center_3d: Ellipsoid center (x, y, z)
 * - axes_3d: Ellipsoid semi-axes [semi_major, semi_minor, semi_height]
 * - rotation_matrix: 3x3 rotation matrix (9 elements)
 * - plane_model: Table plane equation [a, b, c, d]
 * - num_viewpoints: Number of viewpoints to generate
 * - standoff_distance: Distance from ellipsoid surface (meters)
 * - elevation_min: Minimum elevation angle (degrees)
 * - elevation_max: Maximum elevation angle (degrees)
 * - viewpoints_along: Pattern for generating viewpoints ("circular",
 * "major_axis", or "minor_axis")
 * - debug: Enable debug visualization (requires main thread)
 * - object_pcd: Optional object point cloud for visualization
 * - ellipsoid_mesh: Optional ellipsoid mesh for visualization
 *
 * Output Ports:
 * - viewpoints: Array of camera poses (position + orientation quaternion)
 */
class GenerateViewPoints
    : public BT::RosServiceNode<
          scan_and_plan_interfaces::srv::GenerateViewPoints> {
public:
  explicit GenerateViewPoints(const std::string &name,
                              const BT::NodeConfig &conf,
                              const BT::RosNodeParams &params)
      : RosServiceNode<scan_and_plan_interfaces::srv::GenerateViewPoints>(
            name, conf, params) {}

  static BT::PortsList providedPorts() {
    return providedBasicPorts(
        {BT::InputPort<geometry_msgs::msg::Point>("center_3d"),
         BT::InputPort<std::vector<double>>("axes_3d"),
         BT::InputPort<std::vector<double>>("rotation_matrix"),
         BT::InputPort<std::vector<double>>("plane_model"),
         BT::InputPort<int>("num_viewpoints"),
         BT::InputPort<double>("standoff_distance"),
         BT::InputPort<double>("elevation_min"),
         BT::InputPort<double>("elevation_max"),
         BT::InputPort<std::string>(
             "viewpoints_along"), // "circular", "major_axis", or "minor_axis"
         BT::InputPort<bool>("debug"),
         BT::InputPort<sensor_msgs::msg::PointCloud2>("object_pcd"),
         BT::InputPort<shape_msgs::msg::Mesh>("ellipsoid_mesh"),
         BT::OutputPort<std::vector<geometry_msgs::msg::Pose>>("viewpoints")});
  }

  bool setRequest(Request::SharedPtr &request) override;

  BT::NodeStatus
  onResponseReceived(const Response::SharedPtr &response) override;

  virtual BT::NodeStatus onFailure(BT::ServiceNodeErrorCode error) override;
};
