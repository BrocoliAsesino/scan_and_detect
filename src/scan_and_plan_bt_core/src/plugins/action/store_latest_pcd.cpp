#include "plugins/action/store_latest_pcd.hpp"
#include <rclcpp/rclcpp.hpp>

BT::NodeStatus StoreLatestPCD::onTick(
    const std::shared_ptr<sensor_msgs::msg::PointCloud2> &msg) {
  // Check if we received a valid message
  if (!msg) {
    RCLCPP_ERROR(rclcpp::get_logger("StoreLatestPCD"),
                 "No new message received");
    return BT::NodeStatus::FAILURE;
  }

  // Create a new copy of the message to avoid memory issues
  // This ensures the data outlives the shared_ptr lifecycle
  sensor_msgs::msg::PointCloud2 pcd_copy = *msg;
  setOutput("latest_pcd", pcd_copy);

  // Return SUCCESS if the data field is not empty, otherwise FAILURE
  return !pcd_copy.data.empty() ? BT::NodeStatus::SUCCESS
                                : BT::NodeStatus::FAILURE;
}
