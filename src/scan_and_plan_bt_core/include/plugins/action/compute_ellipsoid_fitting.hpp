#pragma once

#include "geometry_msgs/msg/point.hpp"
#include "scan_and_plan_interfaces/srv/compute_ellipsoid_fitting.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "shape_msgs/msg/mesh.hpp"
#include <behaviortree_ros2/bt_service_node.hpp>

#include <string>
#include <vector>

/**
 * @brief Behavior Tree Service Node for ellipsoid fitting to 3D objects.
 *
 * This node calls the ComputeEllipsoidFitting service to:
 * - Extract 2D convex hull from projected object
 * - Fit 2D ellipse to perimeter points
 * - Transform to 3D upper-hemisphere ellipsoid
 * - Generate ellipsoid mesh representation
 *
 * Input Ports:
 * - object_pcd: Filtered object point cloud
 * - plane_model: Table plane equation [a, b, c, d]
 * - height_margin: Additional margin for height estimation (meters)
 * - debug: Enable debug visualization (requires main thread)
 *
 * Output Ports:
 * - center_3d: Ellipsoid center in 3D (x, y, z)
 * - axes_3d: Ellipsoid semi-axes [semi_major, semi_minor, semi_height]
 * - rotation_matrix: 3x3 rotation matrix (flattened to 9 elements)
 * - ellipsoid_mesh: Triangle mesh representation of ellipsoid
 */
class ComputeEllipsoidFitting
    : public BT::RosServiceNode<
          scan_and_plan_interfaces::srv::ComputeEllipsoidFitting> {
public:
  explicit ComputeEllipsoidFitting(const std::string &name,
                                   const BT::NodeConfig &conf,
                                   const BT::RosNodeParams &params)
      : RosServiceNode<scan_and_plan_interfaces::srv::ComputeEllipsoidFitting>(
            name, conf, params) {}

  static BT::PortsList providedPorts() {
    return providedBasicPorts(
        {BT::InputPort<sensor_msgs::msg::PointCloud2>("object_pcd"),
         BT::InputPort<std::vector<double>>("plane_model"),
         BT::InputPort<double>("height_margin"), BT::InputPort<bool>("debug"),
         BT::OutputPort<geometry_msgs::msg::Point>("center_3d"),
         BT::OutputPort<std::vector<double>>("axes_3d"),
         BT::OutputPort<std::vector<double>>("rotation_matrix"),
         BT::OutputPort<shape_msgs::msg::Mesh>("ellipsoid_mesh")});
  }

  bool setRequest(Request::SharedPtr &request) override;

  BT::NodeStatus
  onResponseReceived(const Response::SharedPtr &response) override;

  virtual BT::NodeStatus onFailure(BT::ServiceNodeErrorCode error) override;
};
