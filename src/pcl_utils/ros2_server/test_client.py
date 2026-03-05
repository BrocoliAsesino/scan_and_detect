#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2
from scan_and_plan_interfaces.srv import (
    FilterPCD,
    ComputeEllipsoidFitting,
    GenerateViewPoints,
)
import threading


class PCLUtilsTestClient(Node):
    def __init__(self):
        super().__init__("pcl_utils_test_client")

        # Use reentrant callback groups to allow concurrent callbacks
        self.service_cb_group = ReentrantCallbackGroup()
        self.subscription_cb_group = MutuallyExclusiveCallbackGroup()

        # Create service clients with reentrant callback group
        self.filter_client = self.create_client(
            FilterPCD, "filter_pcd", callback_group=self.service_cb_group
        )
        self.ellipsoid_client = self.create_client(
            ComputeEllipsoidFitting,
            "compute_ellipsoid_fitting",
            callback_group=self.service_cb_group,
        )
        self.viewpoints_client = self.create_client(
            GenerateViewPoints,
            "generate_viewpoints",
            callback_group=self.service_cb_group,
        )

        # Wait for services to be available
        self.get_logger().info("Waiting for services...")
        self.filter_client.wait_for_service(timeout_sec=5.0)
        self.ellipsoid_client.wait_for_service(timeout_sec=5.0)
        self.viewpoints_client.wait_for_service(timeout_sec=5.0)
        self.get_logger().info("All services are available!")

        # Subscribe to pointcloud topic
        self.subscription = self.create_subscription(
            PointCloud2,
            "/camera/depth/color/points",
            self.pointcloud_callback,
            10,
            callback_group=self.subscription_cb_group,
        )

        self.processing_lock = threading.Lock()
        self.processing = False  # Flag to prevent concurrent processing
        self.get_logger().info(
            "Test client initialized. Listening to /camera/depth/color/points"
        )

    def pointcloud_callback(self, msg):
        """Process incoming pointcloud messages"""
        with self.processing_lock:
            if self.processing:
                self.get_logger().info("Already processing, skipping this pointcloud")
                return
            self.processing = True

        # Process in a separate thread to avoid blocking
        threading.Thread(
            target=self.process_pointcloud, args=(msg,), daemon=True
        ).start()

    def process_pointcloud(self, msg):
        """Process pointcloud in a separate thread"""
        self.get_logger().info("=" * 50)
        self.get_logger().info(
            "Received new pointcloud, starting processing pipeline..."
        )

        try:
            # Step 1: Filter the point cloud
            self.get_logger().info("Step 1: Calling filter_pcd service...")
            filter_request = FilterPCD.Request()
            filter_request.input_pcd = msg
            # Use default parameters (empty arrays/zero values will use server defaults)
            filter_request.x_range = [-0.5, 0.5]
            filter_request.y_range = [-0.5, 0.5]
            filter_request.z_range = [0.3, 1.2]
            filter_request.voxel_size = 0.005
            filter_request.plane_distance_threshold = 0.003
            filter_request.plane_ransac_n = 3
            filter_request.plane_num_iterations = 1000
            filter_request.cluster_eps = 0.02
            filter_request.cluster_min_points = 20
            filter_request.print_progress = True

            filter_future = self.filter_client.call_async(filter_request)
            rclpy.spin_until_future_complete(self, filter_future)
            filter_response = filter_future.result()

            if not filter_response.success:
                self.get_logger().error(
                    f"Filter service failed: {filter_response.message}"
                )
                with self.processing_lock:
                    self.processing = False
                return

            self.get_logger().info(
                f"✓ Filter service succeeded: {filter_response.message}"
            )
            self.get_logger().info(f"  Plane model: {filter_response.plane_model}")
            # Step 2: Compute ellipsoid fitting
            self.get_logger().info(
                "Step 2: Calling compute_ellipsoid_fitting service..."
            )
            ellipsoid_request = ComputeEllipsoidFitting.Request()
            ellipsoid_request.object_pcd = filter_response.filtered_pcd
            ellipsoid_request.plane_model = filter_response.plane_model
            ellipsoid_request.height_margin = 0.01
            ellipsoid_request.debug = True  # Set to True to see visualization plots

            ellipsoid_future = self.ellipsoid_client.call_async(ellipsoid_request)
            rclpy.spin_until_future_complete(self, ellipsoid_future)
            ellipsoid_response = ellipsoid_future.result()

            if not ellipsoid_response.success:
                self.get_logger().error(
                    f"Ellipsoid fitting failed: {ellipsoid_response.message}"
                )
                with self.processing_lock:
                    self.processing = False
                return

            self.get_logger().info(
                f"✓ Ellipsoid fitting succeeded: {ellipsoid_response.message}"
            )
            self.get_logger().info(
                f"  Center: [{ellipsoid_response.center_3d.x:.3f}, "
                f"{ellipsoid_response.center_3d.y:.3f}, "
                f"{ellipsoid_response.center_3d.z:.3f}]"
            )
            self.get_logger().info(f"  Axes: {ellipsoid_response.axes_3d}")

            # Step 3: Generate viewpoints
            self.get_logger().info("Step 3: Calling generate_viewpoints service...")
            viewpoints_request = GenerateViewPoints.Request()
            viewpoints_request.center_3d = ellipsoid_response.center_3d
            viewpoints_request.axes_3d = ellipsoid_response.axes_3d
            viewpoints_request.rotation_matrix = ellipsoid_response.rotation_matrix
            viewpoints_request.plane_model = filter_response.plane_model
            viewpoints_request.object_pcd = filter_response.filtered_pcd
            viewpoints_request.ellipsoid_mesh = ellipsoid_response.ellipsoid_mesh
            viewpoints_request.num_viewpoints = 12
            viewpoints_request.standoff_distance = 0.1
            viewpoints_request.elevation_min = 20.0
            viewpoints_request.elevation_max = 70.0
            viewpoints_request.use_minor_axis = False
            viewpoints_request.debug = True  # Set to True to see visualization

            viewpoints_future = self.viewpoints_client.call_async(viewpoints_request)
            rclpy.spin_until_future_complete(self, viewpoints_future)
            viewpoints_response = viewpoints_future.result()

            if not viewpoints_response.success:
                self.get_logger().error(
                    f"Generate viewpoints failed: {viewpoints_response.message}"
                )
                with self.processing_lock:
                    self.processing = False
                return

            self.get_logger().info(
                f"✓ Generate viewpoints succeeded: {viewpoints_response.message}"
            )
            self.get_logger().info(
                f"  Generated {len(viewpoints_response.viewpoints)} viewpoints"
            )

            # Print first few viewpoints as example
            for i, pose in enumerate(viewpoints_response.viewpoints[:3]):
                self.get_logger().info(
                    f"  Viewpoint {i}: pos=[{pose.position.x:.3f}, "
                    f"{pose.position.y:.3f}, {pose.position.z:.3f}]"
                )

            self.get_logger().info("=" * 50)
            self.get_logger().info("✓ ALL SERVICES COMPLETED SUCCESSFULLY!")
            self.get_logger().info("=" * 50)

        except Exception as e:
            self.get_logger().error(f"Error during processing: {str(e)}")
        finally:
            with self.processing_lock:
                self.processing = False


def main(args=None):
    rclpy.init(args=args)
    client = PCLUtilsTestClient()

    # Use MultiThreadedExecutor to allow concurrent callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(client)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
