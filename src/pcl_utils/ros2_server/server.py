#!/usr/bin/env python3

import open3d
import numpy as np

import pcl_utils.open3d_ros_helperV2 as o3d_ros
import pcl_utils.pointcloud_processing as pcd_processing
import pcl_utils.pointcloud_registration as pcd_registration
import pcl_utils.ellipsoid_fitting as ellipsoid_fit

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor


from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point, Pose
from std_srvs.srv import Trigger
from scan_and_plan_interfaces.srv import (
    FilterPCD,
    ComputeEllipsoidFitting,
    GenerateViewPoints,
)

import copy


class FilterPCL(Node):
    def __init__(self):
        super().__init__("pcl_utils_server")
        self.cb_group = ReentrantCallbackGroup()

        # Create services with custom interfaces
        self.create_service(
            FilterPCD,
            "filter_pcd",
            self.filter_pcd_callback,
            callback_group=self.cb_group,
        )
        self.create_service(
            ComputeEllipsoidFitting,
            "compute_ellipsoid_fitting",
            self.ellipsoid_fitting_callback,
            callback_group=self.cb_group,
        )
        self.create_service(
            GenerateViewPoints,
            "generate_viewpoints",
            self.generate_viewpoints_callback,
            callback_group=self.cb_group,
        )

        self.create_timer(0.1, self.timer_callback, callback_group=self.cb_group)

        self.create_service(
            Trigger,
            "store_latest_o3d_pcd",
            self.store_o3d_pcd_callback,
            callback_group=self.cb_group,
        )
        self.create_service(
            Trigger,
            "visualize_pipeline",
            self.visualize_pipeline_callback,
            callback_group=self.cb_group,
        )

        # Publishers for filtered point clouds
        self.filtered_pub = self.create_publisher(
            PointCloud2, "/filtered_pointcloud", 10
        )
        self.object_pub = self.create_publisher(PointCloud2, "/object_pointcloud", 10)

        # Storage for different processing stages
        self.latest_o3d_pcd = open3d.geometry.PointCloud()
        self.raw_pcd = open3d.geometry.PointCloud()
        self.cropped_pcd = open3d.geometry.PointCloud()
        self.downsampled_pcd = open3d.geometry.PointCloud()
        self.denoised_pcd = open3d.geometry.PointCloud()
        self.plane_removed_pcd = open3d.geometry.PointCloud()  # After plane removal
        self.object_pcd = open3d.geometry.PointCloud()  # Final object (largest cluster)
        self.plane_model = None  # Table plane equation [a, b, c, d] from RANSAC

        # Processing parameters - size of the region of interest (ROI) around the object
        self.x_range = [-0.5, 0.5]  # meters
        self.y_range = [-0.5, 0.5]  # meters
        self.z_range = [0.3, 1.2]  # meters
        self.voxel_size = 0.005  # 5mm voxels for downsampling

        # Plane removal parameters
        self.plane_distance_threshold = 0.003  # 3mm tolerance for plane fitting
        self.plane_ransac_n = 3
        self.plane_num_iterations = 1000

        # Clustering parameters
        self.cluster_eps = 0.01  # 10mm neighborhood for clustering
        self.cluster_min_points = 20  # Minimum points to form a cluster

        self.enable_visualization = True  # Set to True to see each step

        # Store latest frame ID for publishing
        self.latest_frame_id = "camera_depth_optical_frame"

        self.get_logger().info(
            "PCL Utils Server initialized. Available services: /filter_pcd, /compute_ellipsoid_fitting, /generate_viewpoints, /visualize_pipeline, /store_latest_o3d_pcd"
        )

    def timer_callback(self):
        # Publish filtered point clouds back to ROS for RViz visualization
        if len(self.denoised_pcd.points) > 0:
            filtered_ros = o3d_ros.o3dpc_to_rospc(
                self.denoised_pcd, frame_id=self.latest_frame_id
            )
            self.filtered_pub.publish(filtered_ros)

        if len(self.object_pcd.points) > 0:
            object_ros = o3d_ros.o3dpc_to_rospc(
                self.object_pcd, frame_id=self.latest_frame_id
            )
            self.object_pub.publish(object_ros)

    def filter_pcd_callback(self, request, response):
        """
        Complete point cloud processing pipeline:
        1. Convert ROS -> Open3D
        2. Pass-through filter (crop to ROI)
        3. Voxel downsampling (reduce density, improve performance)
        4. Statistical outlier removal (remove noise)
        5. Plane segmentation and removal (remove table/ground)
        6. DBSCAN clustering (extract main object)
        """
        try:
            # Store frame ID for publishing
            self.latest_frame_id = request.input_pcd.header.frame_id

            # Use request parameters or defaults
            x_range = request.x_range if len(request.x_range) == 2 else self.x_range
            y_range = request.y_range if len(request.y_range) == 2 else self.y_range
            z_range = request.z_range if len(request.z_range) == 2 else self.z_range
            voxel_size = (
                request.voxel_size if request.voxel_size > 0 else self.voxel_size
            )
            plane_distance_threshold = (
                request.plane_distance_threshold
                if request.plane_distance_threshold > 0
                else self.plane_distance_threshold
            )
            plane_ransac_n = (
                request.plane_ransac_n
                if request.plane_ransac_n > 0
                else self.plane_ransac_n
            )
            plane_num_iterations = (
                request.plane_num_iterations
                if request.plane_num_iterations > 0
                else self.plane_num_iterations
            )
            cluster_eps = (
                request.cluster_eps if request.cluster_eps > 0 else self.cluster_eps
            )
            cluster_min_points = (
                request.cluster_min_points
                if request.cluster_min_points > 0
                else self.cluster_min_points
            )

            # Step 1: Convert ROS PointCloud2 to Open3D
            self.get_logger().info(
                "Received point cloud with %d points"
                % (request.input_pcd.width * request.input_pcd.height)
            )
            self.raw_pcd = o3d_ros.ros2pc_to_o3dpc(request.input_pcd)

            if len(self.raw_pcd.points) == 0:
                response.success = False
                response.message = "Received empty point cloud!"
                self.get_logger().warn(response.message)
                return response

            # Step 2: Apply pass-through filter to focus on region of interest
            self.get_logger().info(
                "Step 1: Raw point cloud has %d points" % len(self.raw_pcd.points)
            )
            self.cropped_pcd = o3d_ros.apply_pass_through_filter(
                self.raw_pcd, x_range, y_range, z_range
            )
            self.get_logger().info(
                "Step 2: After pass-through filter: %d points"
                % len(self.cropped_pcd.points)
            )

            if len(self.cropped_pcd.points) < 100:
                response.success = False
                response.message = (
                    "Too few points after cropping! Check your filter ranges."
                )
                self.get_logger().warn(response.message)
                return response

            # Step 3: Voxel downsampling for efficiency
            # Reduces point density while preserving structure
            self.downsampled_pcd, _ = pcd_registration.prepare_point_cloud_features(
                self.cropped_pcd, voxel_size=voxel_size
            )
            self.get_logger().info(
                "Step 3: After voxel downsampling: %d points"
                % len(self.downsampled_pcd.points)
            )

            # Step 4: Statistical outlier removal to remove noise
            # Removes points that are far from their neighbors (sensor noise, flying pixels)
            self.denoised_pcd = pcd_processing.remove_statistical_outliers(
                self.downsampled_pcd
            )
            self.get_logger().info(
                "Step 4: After denoising: %d points" % len(self.denoised_pcd.points)
            )

            # Step 5: Remove the dominant plane (table/ground)
            # Uses RANSAC to detect and remove the table plane
            self.plane_removed_pcd, plane_model = (
                pcd_processing.remove_plane_from_point_cloud(
                    self.denoised_pcd,
                    distance_threshold=plane_distance_threshold,
                    ransac_n=plane_ransac_n,
                    num_iterations=plane_num_iterations,
                )
            )

            # Store plane model for ellipsoid fitting
            self.plane_model = np.array(plane_model)
            self.get_logger().info(
                "Step 5: After plane removal: %d points (plane: [%.3f, %.3f, %.3f, %.3f])"
                % (
                    len(self.plane_removed_pcd.points),
                    plane_model[0],
                    plane_model[1],
                    plane_model[2],
                    plane_model[3],
                )
            )

            # Step 6: DBSCAN clustering to keep only the largest cluster (the main object)
            # Clusters are well separated, so use looser parameters
            if len(self.plane_removed_pcd.points) > cluster_min_points:
                labels = np.array(
                    pcd_processing.apply_dbscan_clustering(
                        self.plane_removed_pcd,
                        eps=cluster_eps,
                        min_points=cluster_min_points,
                        print_progress=False,
                        nbre_de_cluster_retour=0,
                        seuil=800,
                        type_return="labels",
                    )
                )

                if len(labels) > 0 and labels.max() >= 0:
                    # Find the largest cluster
                    unique_labels, counts = np.unique(
                        labels[labels >= 0], return_counts=True
                    )
                    if len(unique_labels) > 0:
                        largest_cluster_label = unique_labels[np.argmax(counts)]
                        largest_cluster_indices = np.where(
                            labels == largest_cluster_label
                        )[0]
                        self.object_pcd = self.plane_removed_pcd.select_by_index(
                            largest_cluster_indices
                        )
                        self.get_logger().info(
                            "Step 6: After clustering: kept largest cluster with %d points (removed %d other clusters)"
                            % (len(self.object_pcd.points), len(unique_labels) - 1)
                        )
                    else:
                        self.object_pcd = self.plane_removed_pcd
                        self.get_logger().warn(
                            "Step 6: No valid clusters found, keeping all points"
                        )
                else:
                    self.object_pcd = self.plane_removed_pcd
                    self.get_logger().warn(
                        "Step 6: Clustering failed, keeping all points"
                    )
            else:
                self.object_pcd = self.plane_removed_pcd
                self.get_logger().warn(
                    "Step 6: Too few points for clustering, keeping all points"
                )

            self.get_logger().info(
                "Final: Object has %d points" % len(self.object_pcd.points)
            )

            # Store the final result
            self.latest_o3d_pcd = copy.deepcopy(self.object_pcd)

            # Populate response - convert filtered object back to ROS PointCloud2
            response.filtered_pcd = o3d_ros.o3dpc_to_rospc(
                self.object_pcd, frame_id=self.latest_frame_id
            )
            response.plane_model = plane_model
            response.success = True
            response.message = "Point cloud filtering completed successfully"

            return response

        except Exception as e:
            response.success = False
            response.message = "Error in point cloud processing: %s" % str(e)
            self.get_logger().error(response.message)
            return response

    def ellipsoid_fitting_callback(self, request, response):
        """
        Service to perform ellipsoid fitting on the point cloud and generate viewpoints
        """
        self.get_logger().info("Ellipsoid fitting service called")

        try:
            # Convert ROS PointCloud2 to Open3D
            object_pcd = o3d_ros.ros2pc_to_o3dpc(request.object_pcd)

            if len(object_pcd.points) == 0:
                response.success = False
                response.message = (
                    "No object point cloud available (empty point cloud)."
                )
                return response

            if len(request.plane_model) != 4:
                response.success = False
                response.message = (
                    "Invalid plane model. Expected 4 values [a, b, c, d]."
                )
                return response

            plane_model = np.array(request.plane_model)
            height_margin = request.height_margin if request.height_margin > 0 else 0.01

            """
            Main pipeline to fit an ellipsoid to the object point cloud and visualize results.
            """
            # Step 1: Extract 2D perimeter from object projected onto table plane
            perimeter_2d = ellipsoid_fit.extract_2d_convex_hull(object_pcd, plane_model)
            if request.debug:
                ellipsoid_fit.plot_2d_projection(object_pcd, plane_model, perimeter_2d)

            # Step 2: Fit 2D ellipse to the perimeter
            center_2d, axes_2d, angle_2d = ellipsoid_fit.fit_ellipse_to_2d_points(
                perimeter_2d
            )
            if request.debug:
                ellipsoid_fit.plot_ellipse_fit(
                    perimeter_2d, center_2d, axes_2d, angle_2d
                )

            # Step 3: Transform 2D ellipse parameters to 3D ellipsoid
            center_3d, axes_3d, rotation_matrix = ellipsoid_fit.convert_ellipse_to_3d(
                center_2d, axes_2d, angle_2d, plane_model, object_pcd, height_margin
            )

            # Step 4: Create the ellipsoid mesh
            # returns a o3d.geometry.TriangleMesh - could be returned in the response
            ellipsoid_mesh_o3d = ellipsoid_fit.create_upper_ellipsoid_mesh(
                center_3d, axes_3d, rotation_matrix, resolution=20
            )
            if request.debug:
                ellipsoid_fit.display_object_with_ellipsoid(
                    object_pcd, ellipsoid_mesh_o3d, center_3d, axes_3d
                )

            # Populate response
            response.center_3d = Point(
                x=float(center_3d[0]), y=float(center_3d[1]), z=float(center_3d[2])
            )
            response.axes_3d = [float(axes_3d[0]), float(axes_3d[1]), float(axes_3d[2])]
            response.rotation_matrix = rotation_matrix.flatten().tolist()
            response.ellipsoid_mesh = pcd_processing.convert_mesh_to_ros_shape_msg(
                ellipsoid_mesh_o3d
            )
            response.success = True
            response.message = "Ellipsoid fitting completed successfully"
            self.get_logger().info(response.message)

            return response

        except Exception as e:
            response.success = False
            response.message = "Error during ellipsoid fitting: %s" % str(e)
            self.get_logger().error(response.message)
            return response

    def generate_viewpoints_callback(self, request, response):
        """
        Service to generate camera viewpoints around an ellipsoid
        """
        self.get_logger().info("Generate viewpoints service called")

        try:
            # Parse ellipsoid parameters from request
            center_3d = np.array(
                [request.center_3d.x, request.center_3d.y, request.center_3d.z]
            )
            axes_3d = np.array(request.axes_3d)
            rotation_matrix = np.array(request.rotation_matrix).reshape(3, 3)
            plane_model = np.array(request.plane_model)

            # Use request parameters or defaults
            standoff_distance = (
                request.standoff_distance if request.standoff_distance > 0 else 0.1
            )
            elevation_range = (
                (request.elevation_min, request.elevation_max)
                if request.elevation_max > request.elevation_min
                else (20, 70)
            )
            num_viewpoints = (
                request.num_viewpoints if request.num_viewpoints > 0 else 12
            )

            # Parse optional visualization objects
            object_pcd = None
            if request.object_pcd and request.object_pcd.width > 0:
                object_pcd = o3d_ros.ros2pc_to_o3dpc(request.object_pcd)

            # Note: ellipsoid_mesh would need proper mesh message type
            ellipsoid_mesh = None
            if request.ellipsoid_mesh and request.ellipsoid_mesh.triangles:
                ellipsoid_mesh = pcd_processing.convert_ros_shape_msg_to_mesh(
                    request.ellipsoid_mesh
                )

            # Generate viewpoints based on method
            if request.use_minor_axis:
                view_points = (
                    ellipsoid_fit.generate_camera_viewpoints_along_principal_axis(
                        center_3d,
                        axes_3d,
                        rotation_matrix,
                        plane_model,
                        num_viewpoints=num_viewpoints,
                        standoff_distance=standoff_distance,
                        use_minor_axis=True,
                    )
                )
            else:
                view_points = ellipsoid_fit.generate_camera_viewpoints_around_ellipsoid(
                    center_3d,
                    axes_3d,
                    rotation_matrix,
                    num_viewpoints=num_viewpoints,
                    standoff_distance=standoff_distance,
                    elevation_range=elevation_range,
                )

            # Optional: visualize if debug is enabled and objects are provided
            if request.debug and object_pcd is not None:
                ellipsoid_fit.visualize_viewpoints_around_object(
                    pcd=object_pcd,
                    ellipsoid_mesh=ellipsoid_mesh,
                    viewpoints=view_points,
                    center_3d=center_3d,
                    num_viewpoints=num_viewpoints,
                    standoff_distance=standoff_distance,
                    elevation_range=(-90, 90),
                    show_view_lines=True,
                    camera_frame_size=0.05,
                )

            # Populate response with generated viewpoints
            # Convert viewpoints to Pose messages
            response.viewpoints = []
            for position, quaternion in view_points:
                pose = Pose()
                pose.position.x = float(position[0])
                pose.position.y = float(position[1])
                pose.position.z = float(position[2])
                pose.orientation.x = float(quaternion[0])
                pose.orientation.y = float(quaternion[1])
                pose.orientation.z = float(quaternion[2])
                pose.orientation.w = float(quaternion[3])
                response.viewpoints.append(pose)

            response.success = True
            response.message = "Generated %d viewpoints successfully" % len(view_points)
            self.get_logger().info(response.message)

            return response

        except Exception as e:
            response.success = False
            response.message = "Error during viewpoint generation: %s" % str(e)
            self.get_logger().error(response.message)
            return response

    ##############################################################################################
    def visualize_stages(self, pcd_list, titles, colors=None, sequential=True):
        """
        Visualize multiple point clouds either sequentially or side by side
        Args:
            pcd_list: List of Open3D point clouds
            titles: List of titles for each point cloud
            colors: List of RGB colors (0-1 range) for each point cloud
            sequential: If True, show one at a time. If False, show all side by side
        """
        if colors is None:
            colors = [
                [1, 0, 0],  # Red
                [0, 1, 0],  # Green
                [0, 0, 1],  # Blue
                [1, 1, 0],  # Yellow
                [1, 0, 1],  # Magenta
                [0, 1, 1],  # Cyan
            ]

        if sequential:
            # Show each point cloud one at a time with its title
            for i, (pcd, title) in enumerate(zip(pcd_list, titles)):
                if len(pcd.points) == 0:
                    self.get_logger().warn("Skipping %s: empty point cloud" % title)
                    continue

                pcd_copy = copy.deepcopy(pcd)
                pcd_copy.paint_uniform_color(colors[i % len(colors)])

                # Add coordinate frame
                coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1, origin=[0, 0, 0]
                )

                # Log and display
                self.get_logger().info(
                    "Stage %d/%d: %s with %d points"
                    % (i + 1, len(pcd_list), title, len(pcd.points))
                )

                # Window name includes stage info
                window_name = "Stage %d/%d: %s (%d points) - Press Q to continue" % (
                    i + 1,
                    len(pcd_list),
                    title,
                    len(pcd.points),
                )

                open3d.visualization.draw_geometries(
                    [pcd_copy, coord_frame],
                    window_name=window_name,
                    width=1280,
                    height=720,
                )
        else:
            # Show all side by side
            colored_pcds = []
            offset = 0.0
            max_dimension = 0.0

            # First pass: find the maximum dimension for consistent spacing
            for pcd in pcd_list:
                if len(pcd.points) > 0:
                    bbox = pcd.get_axis_aligned_bounding_box()
                    dimension = bbox.max_bound[0] - bbox.min_bound[0]
                    max_dimension = max(max_dimension, dimension)

            # Use consistent spacing based on the largest cloud
            spacing = max_dimension + 0.2

            for i, (pcd, title) in enumerate(zip(pcd_list, titles)):
                if len(pcd.points) == 0:
                    continue

                pcd_copy = copy.deepcopy(pcd)

                # Center each point cloud at its origin before offsetting
                bbox = pcd_copy.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                pcd_copy.translate(-center)  # Center at origin

                # Color the point cloud
                pcd_copy.paint_uniform_color(colors[i % len(colors)])

                # Offset horizontally with consistent spacing
                pcd_copy.translate([offset, 0, 0])
                offset += spacing

                colored_pcds.append(pcd_copy)
                self.get_logger().info("%s: %d points" % (title, len(pcd.points)))

            # Add coordinate frame for reference at each position
            for i in range(len(pcd_list)):
                if len(pcd_list[i].points) > 0:
                    coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.05, origin=[i * spacing, 0, 0]
                    )
                    colored_pcds.append(coord_frame)

            open3d.visualization.draw_geometries(
                colored_pcds,
                window_name="Point Cloud Processing Pipeline - All Stages",
                width=1920,
                height=1080,
            )

    def visualize_pipeline_callback(self, request, response):
        """
        Service to visualize all processing stages side by side
        """
        self.get_logger().info("Visualizing processing pipeline...")

        if len(self.raw_pcd.points) == 0:
            response.success = False
            response.message = (
                "No point cloud data available. Wait for a message first."
            )
            return response

        pcd_list = [
            self.raw_pcd,
            self.cropped_pcd,
            self.downsampled_pcd,
            self.denoised_pcd,
            self.plane_removed_pcd,
            self.object_pcd,
        ]

        titles = [
            "Raw",
            "Cropped",
            "Downsampled",
            "Denoised",
            "Plane Removed",
            "Final Object (Largest Cluster)",
        ]

        self.visualize_stages(pcd_list, titles)

        response.success = True
        response.message = "Visualization complete"
        return response

    def store_o3d_pcd_callback(self, request, response):
        """
        Service to save the processed point cloud to disk
        """
        self.get_logger().info("Store PCD service called")

        if len(self.latest_o3d_pcd.points) == 0:
            response.success = False
            response.message = "No point cloud data to save"
            return response

        try:
            # Save all processing stages
            open3d.io.write_point_cloud("raw_pcd.pcd", self.raw_pcd)
            open3d.io.write_point_cloud("cropped_pcd.pcd", self.cropped_pcd)
            open3d.io.write_point_cloud("downsampled_pcd.pcd", self.downsampled_pcd)
            open3d.io.write_point_cloud("denoised_pcd.pcd", self.denoised_pcd)
            open3d.io.write_point_cloud("plane_removed_pcd.pcd", self.plane_removed_pcd)
            open3d.io.write_point_cloud("object_pcd.pcd", self.object_pcd)
            open3d.io.write_point_cloud("latest_o3d_pcd_copy.pcd", self.latest_o3d_pcd)

            response.success = True
            response.message = "All PCD files saved successfully (raw, cropped, downsampled, denoised, plane_removed, object, latest)"
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = "Failed to save PCD: %s" % str(e)
            self.get_logger().error(response.message)

        return response


def main():
    rclpy.init()
    node = FilterPCL()
    exec = MultiThreadedExecutor(num_threads=2)
    exec.add_node(node)
    try:
        exec.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
