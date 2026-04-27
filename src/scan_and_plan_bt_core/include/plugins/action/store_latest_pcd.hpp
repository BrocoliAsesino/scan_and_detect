#pragma once

#include <behaviortree_ros2/bt_topic_sub_node.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

class StoreLatestPCD
    : public BT::RosTopicSubNode<sensor_msgs::msg::PointCloud2> {
public:
  explicit StoreLatestPCD(const std::string &name, const BT::NodeConfig &conf,
                          const BT::RosNodeParams &params)
      : RosTopicSubNode<sensor_msgs::msg::PointCloud2>(name, conf, params) {}

  static BT::PortsList providedPorts() {
    return providedBasicPorts({BT::OutputPort<sensor_msgs::msg::PointCloud2>(
        "latest_pcd", "Latest point cloud data")});
  }

  // Called on every tick with the last message received
  BT::NodeStatus
  onTick(const std::shared_ptr<sensor_msgs::msg::PointCloud2> &msg) override;
};
