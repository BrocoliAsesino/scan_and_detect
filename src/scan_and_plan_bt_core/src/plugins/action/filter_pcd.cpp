#include "plugins/action/filter_pcd.hpp"
#include <rclcpp/rclcpp.hpp>

bool FilterPCD::setRequest(Request::SharedPtr &request) {
  // Get input point cloud
  auto input_pcd = getInput<sensor_msgs::msg::PointCloud2>("input_pcd");
  if (!input_pcd) {
    RCLCPP_ERROR(logger(), "FilterPCD: Failed to get input_pcd");
    return false;
  }
  request->input_pcd = input_pcd.value();

  // Get optional filter parameters (use service defaults if not provided)
  // Read as std::vector<double> (matching the registered port type) to avoid
  // missing convertFromString<std::vector<float>> specialization in BT.
  auto x_range = getInput<std::vector<double>>("x_range");
  if (x_range) {
    request->x_range.assign(x_range.value().begin(), x_range.value().end());
  }
  auto y_range = getInput<std::vector<double>>("y_range");
  if (y_range) {
    request->y_range.assign(y_range.value().begin(), y_range.value().end());
  }
  auto z_range = getInput<std::vector<double>>("z_range");
  if (z_range) {
    request->z_range.assign(z_range.value().begin(), z_range.value().end());
  }
  getInput("voxel_size", request->voxel_size);
  getInput("plane_distance_threshold", request->plane_distance_threshold);
  getInput("plane_ransac_n", request->plane_ransac_n);
  getInput("plane_num_iterations", request->plane_num_iterations);
  getInput("cluster_eps", request->cluster_eps);
  getInput("cluster_min_points", request->cluster_min_points);
  getInput("print_progress", request->print_progress);

  return true;
}

BT::NodeStatus
FilterPCD::onResponseReceived(const Response::SharedPtr &response) {
  if (!response->success) {
    RCLCPP_ERROR(logger(), "FilterPCD service failed: %s",
                 response->message.c_str());
    return BT::NodeStatus::FAILURE;
  }

  // Create fully independent deep copies
  sensor_msgs::msg::PointCloud2 filtered_pcd_copy;
  filtered_pcd_copy.header = response->filtered_pcd.header;
  filtered_pcd_copy.height = response->filtered_pcd.height;
  filtered_pcd_copy.width = response->filtered_pcd.width;
  filtered_pcd_copy.fields = response->filtered_pcd.fields;
  filtered_pcd_copy.is_bigendian = response->filtered_pcd.is_bigendian;
  filtered_pcd_copy.point_step = response->filtered_pcd.point_step;
  filtered_pcd_copy.row_step = response->filtered_pcd.row_step;
  filtered_pcd_copy.is_dense = response->filtered_pcd.is_dense;
  // Explicitly copy the data vector to ensure deep copy
  filtered_pcd_copy.data.assign(response->filtered_pcd.data.begin(),
                                response->filtered_pcd.data.end());

  std::vector<double> plane_model_copy(response->plane_model.begin(),
                                       response->plane_model.end());

  // Set output ports with the copies
  setOutput("filtered_pcd", filtered_pcd_copy);
  setOutput("plane_model", plane_model_copy);

  RCLCPP_INFO(logger(), "FilterPCD succeeded: %s", response->message.c_str());
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus FilterPCD::onFailure(BT::ServiceNodeErrorCode error) {
  RCLCPP_ERROR(logger(), "FilterPCD service call failed with error: %d",
               static_cast<int>(error));
  return BT::NodeStatus::FAILURE;
}
