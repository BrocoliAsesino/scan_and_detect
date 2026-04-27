#include "plugins/action/compute_ellipsoid_fitting.hpp"
#include <rclcpp/rclcpp.hpp>

bool ComputeEllipsoidFitting::setRequest(Request::SharedPtr &request) {
  // Get required inputs
  auto object_pcd = getInput<sensor_msgs::msg::PointCloud2>("object_pcd");
  if (!object_pcd) {
    RCLCPP_ERROR(logger(), "ComputeEllipsoidFitting: Failed to get object_pcd");
    return false;
  }

  // Deep copy PointCloud2 to avoid shared memory issues
  auto &pcd_val = object_pcd.value();
  request->object_pcd.header = pcd_val.header;
  request->object_pcd.height = pcd_val.height;
  request->object_pcd.width = pcd_val.width;
  request->object_pcd.fields = pcd_val.fields;
  request->object_pcd.is_bigendian = pcd_val.is_bigendian;
  request->object_pcd.point_step = pcd_val.point_step;
  request->object_pcd.row_step = pcd_val.row_step;
  request->object_pcd.is_dense = pcd_val.is_dense;
  request->object_pcd.data.assign(pcd_val.data.begin(), pcd_val.data.end());

  auto plane_model = getInput<std::vector<double>>("plane_model");
  if (!plane_model || plane_model.value().size() != 4) {
    RCLCPP_ERROR(logger(), "ComputeEllipsoidFitting: Failed to get valid "
                           "plane_model (must have 4 elements)");
    return false;
  }
  // Convert std::vector to std::array
  std::copy_n(plane_model.value().begin(), 4, request->plane_model.begin());

  // Get optional parameters
  getInput("height_margin", request->height_margin);
  getInput("debug", request->debug);

  return true;
}

BT::NodeStatus ComputeEllipsoidFitting::onResponseReceived(
    const Response::SharedPtr &response) {
  if (!response->success) {
    RCLCPP_ERROR(logger(), "ComputeEllipsoidFitting service failed: %s",
                 response->message.c_str());
    return BT::NodeStatus::FAILURE;
  }

  // Create deep copies of all outputs
  geometry_msgs::msg::Point center_copy = response->center_3d;
  std::vector<double> axes_copy(response->axes_3d.begin(),
                                response->axes_3d.end());
  std::vector<double> rotation_copy(response->rotation_matrix.begin(),
                                    response->rotation_matrix.end());

  // Deep copy ellipsoid mesh
  shape_msgs::msg::Mesh mesh_copy;
  mesh_copy.triangles.reserve(response->ellipsoid_mesh.triangles.size());
  for (const auto &tri : response->ellipsoid_mesh.triangles) {
    mesh_copy.triangles.push_back(tri);
  }
  mesh_copy.vertices.reserve(response->ellipsoid_mesh.vertices.size());
  for (const auto &vert : response->ellipsoid_mesh.vertices) {
    mesh_copy.vertices.push_back(vert);
  }

  // Set output ports with copies
  setOutput("center_3d", center_copy);
  setOutput("axes_3d", axes_copy);
  setOutput("rotation_matrix", rotation_copy);
  setOutput("ellipsoid_mesh", mesh_copy);

  RCLCPP_INFO(
      logger(), "ComputeEllipsoidFitting succeeded: Center=[%.3f, %.3f, %.3f]",
      response->center_3d.x, response->center_3d.y, response->center_3d.z);
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus
ComputeEllipsoidFitting::onFailure(BT::ServiceNodeErrorCode error) {
  RCLCPP_ERROR(logger(),
               "ComputeEllipsoidFitting service call failed with error: %d",
               static_cast<int>(error));
  return BT::NodeStatus::FAILURE;
}
