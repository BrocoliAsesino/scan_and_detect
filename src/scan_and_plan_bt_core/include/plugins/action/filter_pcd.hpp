#pragma once

#include "scan_and_plan_interfaces/srv/filter_pcd.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <behaviortree_ros2/bt_service_node.hpp>

#include <string>
#include <vector>

/**
 * @brief Behavior Tree Service Node for point cloud filtering pipeline.
 *
 * This node calls the FilterPCD service to perform complete point cloud
 * processing:
 * - Pass-through filtering (ROI cropping)
 * - Voxel downsampling
 * - Statistical outlier removal
 * - Plane segmentation and removal
 * - DBSCAN clustering
 *
 * Input Ports:
 * - input_pcd: Raw input point cloud
 * - x_range: X-axis filtering range [min, max] (meters)
 * - y_range: Y-axis filtering range [min, max] (meters)
 * - z_range: Z-axis filtering range [min, max] (meters)
 * - voxel_size: Voxel size for downsampling (meters)
 * - plane_distance_threshold: Distance threshold for plane fitting (meters)
 * - plane_ransac_n: Number of points for RANSAC
 * - plane_num_iterations: Number of RANSAC iterations
 * - cluster_eps: DBSCAN epsilon parameter (meters)
 * - cluster_min_points: Minimum points per cluster
 * - print_progress: Enable progress logging
 *
 * Output Ports:
 * - filtered_pcd: Filtered object point cloud
 * - plane_model: Detected plane equation [a, b, c, d]
 */
class FilterPCD
    : public BT::RosServiceNode<scan_and_plan_interfaces::srv::FilterPCD> {
public:
  explicit FilterPCD(const std::string &name, const BT::NodeConfig &conf,
                     const BT::RosNodeParams &params)
      : RosServiceNode<scan_and_plan_interfaces::srv::FilterPCD>(name, conf,
                                                                 params) {}

  static BT::PortsList providedPorts() {
    return providedBasicPorts(
        {BT::InputPort<sensor_msgs::msg::PointCloud2>("input_pcd"),
         BT::InputPort<std::vector<double>>("x_range"),
         BT::InputPort<std::vector<double>>("y_range"),
         BT::InputPort<std::vector<double>>("z_range"),
         BT::InputPort<double>("voxel_size"),
         BT::InputPort<double>("plane_distance_threshold"),
         BT::InputPort<int>("plane_ransac_n"),
         BT::InputPort<int>("plane_num_iterations"),
         BT::InputPort<double>("cluster_eps"),
         BT::InputPort<int>("cluster_min_points"),
         BT::InputPort<bool>("print_progress"),
         BT::OutputPort<sensor_msgs::msg::PointCloud2>("filtered_pcd"),
         BT::OutputPort<std::vector<double>>("plane_model")});
  }

  bool setRequest(Request::SharedPtr &request) override;

  BT::NodeStatus
  onResponseReceived(const Response::SharedPtr &response) override;

  virtual BT::NodeStatus onFailure(BT::ServiceNodeErrorCode error) override;
};
