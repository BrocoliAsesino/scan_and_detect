#include "plugins/action/get_viewpoints.hpp"
#include <rclcpp/rclcpp.hpp>

bool GenerateViewPoints::setRequest(Request::SharedPtr &request) {
  // Get required inputs
  auto center_3d = getInput<geometry_msgs::msg::Point>("center_3d");
  if (!center_3d) {
    RCLCPP_ERROR(logger(), "GenerateViewPoints: Failed to get center_3d");
    return false;
  }
  request->center_3d = center_3d.value();

  auto axes_3d = getInput<std::vector<double>>("axes_3d");
  if (!axes_3d || axes_3d.value().size() != 3) {
    RCLCPP_ERROR(logger(), "GenerateViewPoints: Failed to get valid axes_3d "
                           "(must have 3 elements)");
    return false;
  }
  // Convert std::vector<double> to std::array<float, 3>
  for (size_t i = 0; i < 3; ++i) {
    request->axes_3d[i] = static_cast<float>(axes_3d.value()[i]);
  }

  auto rotation_matrix = getInput<std::vector<double>>("rotation_matrix");
  if (!rotation_matrix || rotation_matrix.value().size() != 9) {
    RCLCPP_ERROR(logger(), "GenerateViewPoints: Failed to get valid "
                           "rotation_matrix (must have 9 elements)");
    return false;
  }
  // Convert std::vector<double> to std::array<float, 9>
  for (size_t i = 0; i < 9; ++i) {
    request->rotation_matrix[i] =
        static_cast<float>(rotation_matrix.value()[i]);
  }

  auto plane_model = getInput<std::vector<double>>("plane_model");
  if (!plane_model || plane_model.value().size() != 4) {
    RCLCPP_ERROR(logger(), "GenerateViewPoints: Failed to get valid "
                           "plane_model (must have 4 elements)");
    return false;
  }
  // Convert std::vector to std::array
  std::copy_n(plane_model.value().begin(), 4, request->plane_model.begin());

  // Get optional parameters with defaults
  getInput("num_viewpoints", request->num_viewpoints);
  getInput("standoff_distance", request->standoff_distance);
  getInput("elevation_min", request->elevation_min);
  getInput("elevation_max", request->elevation_max);
  getInput("viewpoints_along", request->viewpoints_along);
  getInput("debug", request->debug);

  // Optional visualization inputs (deep copy if provided)
  auto opt_pcd = getInput<sensor_msgs::msg::PointCloud2>("object_pcd");
  if (opt_pcd) {
    auto &pcd_val = opt_pcd.value();
    request->object_pcd.header = pcd_val.header;
    request->object_pcd.height = pcd_val.height;
    request->object_pcd.width = pcd_val.width;
    request->object_pcd.fields = pcd_val.fields;
    request->object_pcd.is_bigendian = pcd_val.is_bigendian;
    request->object_pcd.point_step = pcd_val.point_step;
    request->object_pcd.row_step = pcd_val.row_step;
    request->object_pcd.is_dense = pcd_val.is_dense;
    request->object_pcd.data.assign(pcd_val.data.begin(), pcd_val.data.end());
  }

  auto opt_mesh = getInput<shape_msgs::msg::Mesh>("ellipsoid_mesh");
  if (opt_mesh) {
    auto &mesh_val = opt_mesh.value();
    request->ellipsoid_mesh.triangles.reserve(mesh_val.triangles.size());
    for (const auto &tri : mesh_val.triangles) {
      request->ellipsoid_mesh.triangles.push_back(tri);
    }
    request->ellipsoid_mesh.vertices.reserve(mesh_val.vertices.size());
    for (const auto &vert : mesh_val.vertices) {
      request->ellipsoid_mesh.vertices.push_back(vert);
    }
  }

  return true;
}

BT::NodeStatus
GenerateViewPoints::onResponseReceived(const Response::SharedPtr &response) {
  if (!response->success) {
    RCLCPP_ERROR(logger(), "GenerateViewPoints service failed: %s",
                 response->message.c_str());
    return BT::NodeStatus::FAILURE;
  }

  // Create deep copy of viewpoints vector with explicit field copying
  std::vector<geometry_msgs::msg::Pose> viewpoints_copy;
  viewpoints_copy.reserve(response->viewpoints.size());
  for (const auto &pose_ref : response->viewpoints) {
    geometry_msgs::msg::Pose pose;
    pose.position.x = pose_ref.position.x;
    pose.position.y = pose_ref.position.y;
    pose.position.z = pose_ref.position.z;
    pose.orientation.x = pose_ref.orientation.x;
    pose.orientation.y = pose_ref.orientation.y;
    pose.orientation.z = pose_ref.orientation.z;
    pose.orientation.w = pose_ref.orientation.w;
    viewpoints_copy.push_back(pose);
  }

  // Set output port with copy
  setOutput("viewpoints", viewpoints_copy);

  RCLCPP_INFO(logger(),
              "GenerateViewPoints succeeded: Generated %zu viewpoints",
              response->viewpoints.size());
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus GenerateViewPoints::onFailure(BT::ServiceNodeErrorCode error) {
  RCLCPP_ERROR(logger(),
               "GenerateViewPoints service call failed with error: %d",
               static_cast<int>(error));
  return BT::NodeStatus::FAILURE;
}
