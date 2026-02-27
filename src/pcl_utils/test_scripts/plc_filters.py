#!/usr/bin/env python3

import open3d
import numpy as np

import pcl_utils.open3d_ros_helperV2 as o3d_ros
from pcl_utils.surface_reconstruction_lib import supress_plane, statistical_outlier_removal, db_scan_filter

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor


from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger

import copy

from pcl_utils.surface_playground import extract_perimeter_2d, fit_ellipse_to_perimeter, transform_ellipse_to_3d, create_upper_ellipsoid_mesh, visualize_2d_ellipse_fit, visualize_2d_projection, visualize_object_and_ellipsoid

class FilterPCL(Node):
    def __init__(self):
        super().__init__('test_pcl_filter')
        self.cb_group = ReentrantCallbackGroup()
        
        self.create_subscription(PointCloud2, '/camera/depth/color/points', self.point_cloud_callback, 1, callback_group=self.cb_group)
        self.create_service(Trigger, 'store_latest_o3d_pcd', self.store_o3d_pcd_callback, callback_group=self.cb_group)
        self.create_service(Trigger, 'visualize_pipeline', self.visualize_pipeline_callback, callback_group=self.cb_group)
        
        # Publishers for filtered point clouds
        self.filtered_pub = self.create_publisher(PointCloud2, '/filtered_pointcloud', 10)
        self.object_pub = self.create_publisher(PointCloud2, '/object_pointcloud', 10)

        # Storage for different processing stages
        self.latest_o3d_pcd = open3d.geometry.PointCloud()
        self.raw_pcd = open3d.geometry.PointCloud()
        self.cropped_pcd = open3d.geometry.PointCloud()
        self.downsampled_pcd = open3d.geometry.PointCloud()
        self.denoised_pcd = open3d.geometry.PointCloud()
        self.plane_removed_pcd = open3d.geometry.PointCloud()  # After plane removal
        self.object_pcd = open3d.geometry.PointCloud()  # Final object (largest cluster)
        self.plane_model = None  # Table plane equation [a, b, c, d] from RANSAC
        
        # Processing parameters
        self.x_range = [-0.5, 0.5]  # meters
        self.y_range = [-0.5, 0.5]  # meters
        self.z_range = [0.3, 1.2]   # meters
        self.voxel_size = 0.005  # 5mm voxels for downsampling
        
        # Plane removal parameters
        self.plane_distance_threshold = 0.003  # 3mm tolerance for plane fitting
        self.plane_ransac_n = 3
        self.plane_num_iterations = 1000
        
        # Clustering parameters
        self.cluster_eps = 0.01  # 10mm neighborhood for clustering
        self.cluster_min_points = 20  # Minimum points to form a cluster
        
        self.enable_visualization = True  # Set to True to see each step
        
        self.get_logger().info('FilterPCL node initialized. Call /visualize_pipeline service to see processing stages.')

    def point_cloud_callback(self, msg):
        """
        Complete point cloud processing pipeline:
        1. Convert ROS -> Open3D
        2. Pass-through filter (crop to ROI)
        3. Voxel downsampling (reduce density, improve performance)
        4. Statistical outlier removal (remove noise)
        5. Plane segmentation and removal (remove table/ground)
        """
        try:
            # Step 1: Convert ROS PointCloud2 to Open3D
            self.get_logger().info('Received point cloud with %d points' % (msg.width * msg.height))
            self.raw_pcd = o3d_ros.ros2pc_to_o3dpc(msg)
            
            if len(self.raw_pcd.points) == 0:
                self.get_logger().warn('Received empty point cloud!')
                return
            
            # Step 2: Apply pass-through filter to focus on region of interest
            self.get_logger().info('Step 1: Raw point cloud has %d points' % len(self.raw_pcd.points))
            self.cropped_pcd = o3d_ros.apply_pass_through_filter(
                self.raw_pcd, self.x_range, self.y_range, self.z_range
            )
            self.get_logger().info('Step 2: After pass-through filter: %d points' % len(self.cropped_pcd.points))
            
            if len(self.cropped_pcd.points) < 100:
                self.get_logger().warn('Too few points after cropping! Check your filter ranges.')
                return
            
            # Step 3: Voxel downsampling for efficiency
            # Reduces point density while preserving structure
            self.downsampled_pcd = self.cropped_pcd.voxel_down_sample(voxel_size=self.voxel_size)
            self.get_logger().info('Step 3: After voxel downsampling: %d points' % len(self.downsampled_pcd.points))
            
            # Step 4: Statistical outlier removal to remove noise
            # Removes points that are far from their neighbors (sensor noise, flying pixels)
            self.denoised_pcd = statistical_outlier_removal(self.downsampled_pcd)
            self.get_logger().info('Step 4: After denoising: %d points' % len(self.denoised_pcd.points))
            
            # Step 5: Remove the dominant plane (table/ground)
            # Uses RANSAC to detect and remove the table plane
            plane_model, inliers = self.denoised_pcd.segment_plane(
                distance_threshold=self.plane_distance_threshold,
                ransac_n=self.plane_ransac_n,
                num_iterations=self.plane_num_iterations
            )
            # Store plane model for ellipsoid fitting
            self.plane_model = np.array(plane_model)
            self.plane_removed_pcd = self.denoised_pcd.select_by_index(inliers, invert=True)
            self.get_logger().info('Step 5: After plane removal: %d points (plane: [%.3f, %.3f, %.3f, %.3f])' % 
                                  (len(self.plane_removed_pcd.points), plane_model[0], plane_model[1], plane_model[2], plane_model[3]))
            
            self.filtered_pcd = self.plane_removed_pcd
            
            # Step 6: DBSCAN clustering to keep only the largest cluster (the main object)
            # Clusters are well separated, so use looser parameters
            if len(self.plane_removed_pcd.points) > self.cluster_min_points:
                labels = np.array(self.plane_removed_pcd.cluster_dbscan(
                    eps=0.02,  # 20mm neighborhood - good for well-separated clusters
                    min_points=20,  # Lower threshold since clusters are separated
                    print_progress=False
                ))
                
                if len(labels) > 0 and labels.max() >= 0:
                    # Find the largest cluster
                    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                    if len(unique_labels) > 0:
                        largest_cluster_label = unique_labels[np.argmax(counts)]
                        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
                        self.object_pcd = self.plane_removed_pcd.select_by_index(largest_cluster_indices)
                        self.get_logger().info('Step 6: After clustering: kept largest cluster with %d points (removed %d other clusters)' % 
                                             (len(self.object_pcd.points), len(unique_labels) - 1))
                    else:
                        self.object_pcd = self.plane_removed_pcd
                        self.get_logger().warn('Step 6: No valid clusters found, keeping all points')
                else:
                    self.object_pcd = self.plane_removed_pcd
                    self.get_logger().warn('Step 6: Clustering failed, keeping all points')
            else:
                self.object_pcd = self.plane_removed_pcd
                self.get_logger().warn('Step 6: Too few points for clustering, keeping all points')
            
            self.get_logger().info('Final: Object has %d points' % len(self.object_pcd.points))
            
            # Store the final result
            self.latest_o3d_pcd = copy.deepcopy(self.object_pcd)
            
            # Optional: Estimate normals for the final object (useful for mesh reconstruction later)
            if len(self.object_pcd.points) > 50:
                self.object_pcd.estimate_normals(
                    search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
                )
                self.get_logger().info('Normals estimated for object point cloud')
            
            # Publish filtered point clouds back to ROS for RViz visualization
            if len(self.denoised_pcd.points) > 0:
                filtered_ros = o3d_ros.o3dpc_to_rospc(self.denoised_pcd, frame_id=msg.header.frame_id)
                self.filtered_pub.publish(filtered_ros)
            
            if len(self.object_pcd.points) > 0:
                object_ros = o3d_ros.o3dpc_to_rospc(self.object_pcd, frame_id=msg.header.frame_id)
                self.object_pub.publish(object_ros)
            
            self.get_logger().info('Processing complete! Final object has %d points' % len(self.latest_o3d_pcd.points))
            
            # Step 1: Extract 2D perimeter from object projected onto table plane
            perimeter_2d = extract_perimeter_2d(self.object_pcd, self.plane_model)
            visualize_2d_projection(self.object_pcd, self.plane_model, perimeter_2d)

            # Step 2: Fit 2D ellipse to the perimeter
            center_2d, axes_2d, angle_2d = fit_ellipse_to_perimeter(perimeter_2d)
            visualize_2d_ellipse_fit(perimeter_2d, center_2d, axes_2d, angle_2d)
            # Step 3: Transform 2D ellipse parameters to 3D ellipsoid
            center_3d, axes_3d, rotation_matrix = transform_ellipse_to_3d(
                center_2d, axes_2d, angle_2d, self.plane_model, self.object_pcd
            )
            # visualize_object_and_ellipsoid(self.object_pcd, center_3d, axes_3d, rotation_matrix)

            # Step 4: Create the ellipsoid mesh
            ellipsoid_mesh = create_upper_ellipsoid_mesh(
                center_3d, axes_3d, rotation_matrix, resolution=20
            )
        except Exception as e:
            self.get_logger().error('Error in point cloud processing: %s' % str(e))
        

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
                [1, 0, 0],      # Red
                [0, 1, 0],      # Green
                [0, 0, 1],      # Blue
                [1, 1, 0],      # Yellow
                [1, 0, 1],      # Magenta
                [0, 1, 1],      # Cyan
            ]
        
        if sequential:
            # Show each point cloud one at a time with its title
            for i, (pcd, title) in enumerate(zip(pcd_list, titles)):
                if len(pcd.points) == 0:
                    self.get_logger().warn('Skipping %s: empty point cloud' % title)
                    continue
                
                pcd_copy = copy.deepcopy(pcd)
                pcd_copy.paint_uniform_color(colors[i % len(colors)])
                
                # Add coordinate frame
                coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1, origin=[0, 0, 0]
                )
                
                # Log and display
                self.get_logger().info('Stage %d/%d: %s with %d points' % 
                                      (i+1, len(pcd_list), title, len(pcd.points)))
                
                # Window name includes stage info
                window_name = "Stage %d/%d: %s (%d points) - Press Q to continue" % \
                             (i+1, len(pcd_list), title, len(pcd.points))
                
                open3d.visualization.draw_geometries(
                    [pcd_copy, coord_frame],
                    window_name=window_name,
                    width=1280,
                    height=720
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
                self.get_logger().info('%s: %d points' % (title, len(pcd.points)))
            
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
                height=1080
            )
    
    def visualize_pipeline_callback(self, request, response):
        """
        Service to visualize all processing stages side by side
        """
        self.get_logger().info('Visualizing processing pipeline...')
        
        if len(self.raw_pcd.points) == 0:
            response.success = False
            response.message = 'No point cloud data available. Wait for a message first.'
            return response
        
        pcd_list = [
            self.raw_pcd,
            self.cropped_pcd,
            self.downsampled_pcd,
            self.denoised_pcd,
            self.plane_removed_pcd,
            self.object_pcd
        ]
        
        titles = [
            'Raw',
            'Cropped',
            'Downsampled',
            'Denoised',
            'Plane Removed',
            'Final Object (Largest Cluster)'
        ]
        
        self.visualize_stages(pcd_list, titles)
        
        response.success = True
        response.message = 'Visualization complete'
        return response

    def store_o3d_pcd_callback(self, request, response):
        """
        Service to save the processed point cloud to disk
        """
        self.get_logger().info('Store PCD service called')
        
        if len(self.latest_o3d_pcd.points) == 0:
            response.success = False
            response.message = 'No point cloud data to save'
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
            response.message = 'All PCD files saved successfully (raw, cropped, downsampled, denoised, plane_removed, object, latest)'
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = 'Failed to save PCD: %s' % str(e)
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

if __name__ == '__main__':
    main()