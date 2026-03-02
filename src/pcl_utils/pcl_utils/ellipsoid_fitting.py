#!/usr/bin/env python3
"""
Ellipsoid fitting and viewpoint generation utilities for robotic scanning.

This module provides functions to:
1. Extract 2D perimeter from filtered 3D object point clouds
2. Fit 2D ellipses to object perimeters
3. Generate 3D upper-hemisphere ellipsoids
4. Compute camera viewpoints around objects for robot scanning
"""

import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import ConvexHull
from typing import Tuple, List
import copy
import matplotlib.pyplot as plt


# 2D Projections and Perimeters
def project_point_cloud_to_plane(pcd: o3d.geometry.PointCloud, plane_model: np.ndarray) -> np.ndarray:
    """
    Project 3D point cloud points onto a 2D plane.
    
    Args:
        pcd: Open3D point cloud --> the part that we remove in the plane removal step
        plane_model: Plane equation coefficients [a, b, c, d] where ax + by + cz + d = 0
        
    Returns:
        ndarray: Nx2 array of 2D projected points
    """
    points_3d = np.asarray(pcd.points)
    
    # Extract plane normal and normalize
    normal = plane_model[:3]
    normal = normal / np.linalg.norm(normal)
    
    # Project points onto plane by removing component along normal
    d = plane_model[3]
    distances = np.dot(points_3d, normal) + d
    projected_3d = points_3d - distances[:, np.newaxis] * normal
    
    # Create 2D coordinate system on plane
    # Choose arbitrary perpendicular vectors
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Project to 2D
    points_2d = np.column_stack([
        np.dot(projected_3d, u),
        np.dot(projected_3d, v)
    ])
    
    return points_2d

def extract_2d_convex_hull(pcd: o3d.geometry.PointCloud, plane_model: np.ndarray) -> np.ndarray:
    """
    Extract the 2D perimeter (convex hull) of a 3D object projected onto a plane.
    
    Args:
        pcd: Open3D point cloud of the object
        plane_model: Plane equation coefficients [a, b, c, d]
        
    Returns:
        ndarray: Mx2 array of 2D perimeter points (convex hull vertices)
    """
    # Project to 2D
    points_2d = project_point_cloud_to_plane(pcd, plane_model)
    
    # Compute convex hull
    if len(points_2d) < 5:
        raise ValueError(f"Need at least 5 points for ellipse fitting, got {len(points_2d)}")
    
    hull = ConvexHull(points_2d)
    perimeter_points = points_2d[hull.vertices]
    
    return perimeter_points


# Ellipsoid Fitting and Transformation
def fit_ellipse_to_2d_points(perimeter_points_2d: np.ndarray) -> tuple[np.ndarray, tuple[float, float], float]:
    """
    Fit a 2D ellipse to perimeter points using OpenCV.
    
    Args:
        perimeter_points_2d: Nx2 array of 2D perimeter points
        
    Returns:
        tuple containing:
        - center: (x, y) center of ellipse
        - axes: (major_axis, minor_axis) semi-axis lengths
        - angle: rotation angle in degrees
    """
    if len(perimeter_points_2d) < 5:
        raise ValueError(f"cv2.fitEllipse requires at least 5 points, got {len(perimeter_points_2d)}")
    
    # OpenCV expects points as float32 in shape (N, 1, 2)
    points_cv = perimeter_points_2d.astype(np.float32).reshape(-1, 1, 2)
    
    # Fit ellipse: returns ((cx, cy), (major*2, minor*2), angle)
    ellipse = cv2.fitEllipse(points_cv)
    
    center = np.array(ellipse[0])  # (cx, cy)
    axes = (ellipse[1][0] / 2.0, ellipse[1][1] / 2.0)  # Convert diameter to radius
    angle = ellipse[2]  # Rotation angle in degrees
    
    return center, axes, angle

def convert_ellipse_to_3d(center_2d: np.ndarray, axes_2d: tuple[float, float], angle_2d: float, plane_model: np.ndarray, pcd: o3d.geometry.PointCloud, height_margin: float = 0.01) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    """
    Transform 2D ellipse parameters back to 3D space.
    
    Args:
        center_2d: (x, y) center in 2D plane coordinates
        axes_2d: (major, minor) semi-axis lengths
        angle_2d: rotation angle in degrees
        plane_model: Plane equation [a, b, c, d]
        pcd: Original 3D point cloud (to determine height)
        height_margin: Additional margin to add to the estimated height
        
    Returns:
        tuple containing:
        - center_3d: (x, y, z) center in 3D
        - axes_3d: (semi_major, semi_minor, semi_height) for ellipsoid
        - rotation_matrix: 3x3 rotation matrix for ellipsoid orientation
    """
    points_3d = np.asarray(pcd.points)
    
    # Get plane normal
    normal = plane_model[:3]
    normal = normal / np.linalg.norm(normal)
    
    # Reconstruct 2D coordinate system
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Transform 2D center to 3D
    # Find a point on the plane (use origin projection)
    d = plane_model[3]
    plane_point = -d * normal
    center_3d = plane_point + center_2d[0] * u + center_2d[1] * v
    
    # Estimate object height (along plane normal)
    # Project all points along normal direction
    heights = np.dot(points_3d - center_3d, normal)
    semi_height = max(abs(heights.max()), abs(heights.min())) + height_margin
    
    # Create rotation matrix from 2D angle
    angle_rad = np.radians(angle_2d)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Major axis direction in 3D
    major_dir = cos_a * u + sin_a * v
    minor_dir = -sin_a * u + cos_a * v
    
    # Build 3x3 rotation matrix: columns are [major_axis, minor_axis, normal]
    rotation_matrix = np.column_stack([major_dir, minor_dir, normal])
    
    axes_3d = (axes_2d[0], axes_2d[1], semi_height)
    
    return center_3d, axes_3d, rotation_matrix


def create_upper_ellipsoid_mesh(center_3d: np.ndarray, axes_3d: tuple[float, float, float], rotation_matrix: np.ndarray, resolution: int = 20) -> o3d.geometry.TriangleMesh:
    """
    Create a 3D upper-hemisphere ellipsoid mesh.
    
    Args:
        center_3d: (x, y, z) center of ellipsoid
        axes_3d: (semi_a, semi_b, semi_c) semi-axis lengths
        rotation_matrix: 3x3 rotation matrix for ellipsoid orientation
        resolution: mesh resolution (number of divisions)
        
    Returns:
        Open3D TriangleMesh of upper hemisphere ellipsoid
    """
    # Create unit sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    
    # Get vertices
    vertices = np.asarray(sphere.vertices)
    
    # Keep only upper hemisphere (z >= 0 in local coordinates)
    upper_mask = vertices[:, 2] >= 0
    
    # Scale to ellipsoid
    vertices[:, 0] *= axes_3d[0]  # semi-major axis
    vertices[:, 1] *= axes_3d[1]  # semi-minor axis
    vertices[:, 2] *= axes_3d[2]  # semi-height
    
    # Apply rotation
    vertices = vertices @ rotation_matrix.T
    
    # Translate to center
    vertices += center_3d
    
    # Update mesh
    sphere.vertices = o3d.utility.Vector3dVector(vertices)
    
    # Remove lower hemisphere triangles
    vertices_array = np.asarray(sphere.vertices)
    triangles = np.asarray(sphere.triangles)
    
    # Keep triangles where at least 2 vertices are in upper hemisphere
    # Transform back to check
    vertices_local = (vertices_array - center_3d) @ rotation_matrix
    upper_vertices = vertices_local[:, 2] >= -0.01 * axes_3d[2]  # Small tolerance
    
    valid_triangles = []
    for tri in triangles:
        if np.sum(upper_vertices[tri]) >= 2:
            valid_triangles.append(tri)
    
    sphere.triangles = o3d.utility.Vector3iVector(np.array(valid_triangles))
    
    # Clean up
    sphere.remove_duplicated_vertices()
    sphere.remove_degenerate_triangles()
    sphere.compute_vertex_normals()
    
    return sphere


# Viewpoint Generation
def generate_camera_viewpoints(center_3d: np.ndarray, axes_3d: tuple[float, float, float], rotation_matrix: np.ndarray, num_viewpoints: int = 12, standoff_distance: float = 0.3, elevation_range: tuple[float, float] = (20, 70)) -> List[tuple[np.ndarray, np.ndarray]]:
    """
    Generate camera viewpoints around an ellipsoid for robotic scanning.
    
    Args:
        center_3d: (x, y, z) center of ellipsoid
        axes_3d: (semi_a, semi_b, semi_c) semi-axis lengths
        rotation_matrix: 3x3 rotation matrix
        num_viewpoints: number of viewpoints to generate
        standoff_distance: distance from ellipsoid surface (meters)
        elevation_range: (min_deg, max_deg) elevation angles for viewpoints
        
    Returns:
        List of (position, orientation_quaternion) tuples for each viewpoint
        position: (x, y, z) camera position
        orientation: (x, y, z, w) quaternion looking at center
    """
    viewpoints = []
    
    # Generate uniformly distributed azimuth angles
    azimuth_angles = np.linspace(0, 2 * np.pi, num_viewpoints, endpoint=False)
    
    # Use mid-range elevation
    elevation_rad = np.radians((elevation_range[0] + elevation_range[1]) / 2.0)
    
    for azimuth in azimuth_angles:
        # Parametric point on ellipsoid surface (spherical coordinates)
        local_point = np.array([
            axes_3d[0] * np.sin(elevation_rad) * np.cos(azimuth),
            axes_3d[1] * np.sin(elevation_rad) * np.sin(azimuth),
            axes_3d[2] * np.cos(elevation_rad)
        ])
        
        # Transform to world coordinates
        point_on_surface = rotation_matrix @ local_point + center_3d
        
        # Compute outward normal at this point (gradient of ellipsoid equation)
        local_normal = local_point / (np.array(axes_3d) ** 2)
        local_normal = local_normal / np.linalg.norm(local_normal)
        world_normal = rotation_matrix @ local_normal
        
        # Camera position: offset from surface along normal
        camera_position = point_on_surface + standoff_distance * world_normal
        
        # Compute look-at orientation (camera Z-axis points at center)
        view_direction = center_3d - camera_position
        view_direction = view_direction / np.linalg.norm(view_direction)
        
        # Camera frame: Z forward (pointing at object), Y down, X right
        z_axis = view_direction
        
        # Choose up vector (world Z) and compute right vector
        world_up = np.array([0, 0, 1])
        x_axis = np.cross(world_up, z_axis)
        if np.linalg.norm(x_axis) < 0.001:
            # View direction is vertical, choose different up
            world_up = np.array([0, 1, 0])
            x_axis = np.cross(world_up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        # Build rotation matrix: columns are camera frame axes
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # Convert rotation matrix to quaternion
        quaternion = rotation_matrix_to_quaternion(R)
        
        viewpoints.append((camera_position, quaternion))
    
    return viewpoints

def generate_camera_viewpoints_along_principal_axis(center_3d: np.ndarray, 
                                                    axes_3d: tuple[float, float, float], 
                                                    rotation_matrix: np.ndarray,
                                                    plane_model: np.ndarray,
                                                    num_viewpoints: int = 12, 
                                                    standoff_distance: float = 0.3) -> List[tuple[np.ndarray, np.ndarray]]:
    """
    Generate camera viewpoints along the principal axis, looking straight down at the plane.
    
    Args:
        center_3d: (x, y, z) center of ellipsoid
        axes_3d: (semi_a, semi_b, semi_c) semi-axis lengths
        rotation_matrix: 3x3 rotation matrix
        plane_model: Plane equation coefficients [a, b, c, d] for table plane
        num_viewpoints: number of viewpoints to generate
        standoff_distance: distance from ellipsoid surface (meters)
        
    Returns:
        List of (position, orientation_quaternion) tuples for each viewpoint
        position: (x, y, z) camera position
        orientation: (x, y, z, w) quaternion looking straight down at plane
    """
    viewpoints = []
    
    # Extract and normalize plane normal (pointing up from table)
    plane_normal = plane_model[:3]
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Generate uniformly distributed angles
    azimuth_angles = np.pi / 2  # Fixed azimuth = 90 degrees to align with principal (major) axis
    elevation_angles = np.linspace(0.33 * np.pi, -0.33 * np.pi, num_viewpoints, endpoint=False)
    
    for elevation in elevation_angles:
        # Parametric point on ellipsoid surface (spherical coordinates)
        local_point = np.array([
            axes_3d[0] * np.sin(elevation) * np.cos(azimuth_angles),
            axes_3d[1] * np.sin(elevation) * np.sin(azimuth_angles),
            axes_3d[2] * np.cos(elevation)
        ])
        
        # Transform to world coordinates
        point_on_surface = rotation_matrix @ local_point + center_3d
        
        # Compute outward normal at this point (gradient of ellipsoid equation)
        local_normal = local_point / (np.array(axes_3d) ** 2)
        local_normal = local_normal / np.linalg.norm(local_normal)
        world_normal = rotation_matrix @ local_normal
        
        # Camera position: offset from surface along normal
        camera_position = point_on_surface + standoff_distance * world_normal
        
        # Camera looks straight down toward plane (opposite of plane normal)
        view_direction = -plane_normal
        view_direction = view_direction / np.linalg.norm(view_direction)
        
        # Camera frame: Z forward (pointing at object), Y down, X right
        z_axis = view_direction
        
        # Choose up vector (world Z) and compute right vector
        world_up = np.array([0, 0, 1])
        x_axis = np.cross(world_up, z_axis)
        if np.linalg.norm(x_axis) < 0.001:
            # View direction is vertical, choose different up
            world_up = np.array([0, 1, 0])
            x_axis = np.cross(world_up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        # Build rotation matrix: columns are camera frame axes
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # Convert rotation matrix to quaternion
        quaternion = rotation_matrix_to_quaternion(R)
        
        viewpoints.append((camera_position, quaternion))
    
    return viewpoints


# Visualization
def plot_2d_projection(pcd: o3d.geometry.PointCloud, plane_model: np.ndarray, perimeter_points: np.ndarray = None, title: str = "2D Projection and Perimeter"):
    """
    Visualize 2D projection of 3D point cloud onto plane with optional perimeter.
    
    Args:
        pcd: Original 3D point cloud
        plane_model: Plane equation [a, b, c, d]
        perimeter_points: Optional Nx2 array of perimeter points
        title: Plot title
    """
    # Project all points to 2D
    points_2d = project_point_cloud_to_plane(pcd, plane_model)
    
    plt.figure(figsize=(10, 10))
    
    # Plot all projected points
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='blue', alpha=0.3, s=1, label='Projected points')
    
    # Plot perimeter if provided
    if perimeter_points is not None:
        # Close the perimeter loop for plotting
        perimeter_closed = np.vstack([perimeter_points, perimeter_points[0]])
        plt.plot(perimeter_closed[:, 0], perimeter_closed[:, 1], 'r-', linewidth=2, label='Convex hull perimeter')
        plt.scatter(perimeter_points[:, 0], perimeter_points[:, 1], c='red', s=50, zorder=5, label='Perimeter vertices')
    
    plt.xlabel('U axis (m)')
    plt.ylabel('V axis (m)')
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_ellipse_fit(perimeter_points: np.ndarray, center_2d: np.ndarray, axes_2d: tuple[float, float], angle_2d: float, title: str = "2D Ellipse Fit"):
    """
    Visualize fitted 2D ellipse over perimeter points.
    
    Args:
        perimeter_points: Nx2 array of perimeter points
        center_2d: (x, y) ellipse center
        axes_2d: (major, minor) semi-axis lengths
        angle_2d: Rotation angle in degrees
        title: Plot title
    """
    plt.figure(figsize=(10, 10))
    
    # Plot perimeter points
    perimeter_closed = np.vstack([perimeter_points, perimeter_points[0]])
    plt.plot(perimeter_closed[:, 0], perimeter_closed[:, 1], 'b-', linewidth=2, label='Perimeter')
    plt.scatter(perimeter_points[:, 0], perimeter_points[:, 1], c='blue', s=50, zorder=5)
    
    # Generate ellipse points for visualization
    t = np.linspace(0, 2*np.pi, 100)
    
    # Parametric ellipse in local coordinates
    x_local = axes_2d[0] * np.cos(t)
    y_local = axes_2d[1] * np.sin(t)
    
    # Rotate by angle
    angle_rad = np.radians(angle_2d)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    x_rot = cos_a * x_local - sin_a * y_local + center_2d[0]
    y_rot = sin_a * x_local + cos_a * y_local + center_2d[1]
    
    plt.plot(x_rot, y_rot, 'r-', linewidth=2, label='Fitted ellipse')
    plt.scatter(center_2d[0], center_2d[1], c='red', s=100, marker='x', linewidths=3, zorder=10, label='Center')
    
    # Draw major and minor axes
    major_vec = np.array([axes_2d[0] * cos_a, axes_2d[0] * sin_a])
    minor_vec = np.array([-axes_2d[1] * sin_a, axes_2d[1] * cos_a])
    
    plt.arrow(center_2d[0], center_2d[1], major_vec[0], major_vec[1],
              head_width=0.02, head_length=0.02, fc='green', ec='green', linewidth=2, label='Major axis')
    plt.arrow(center_2d[0], center_2d[1], minor_vec[0], minor_vec[1],
              head_width=0.02, head_length=0.02, fc='orange', ec='orange', linewidth=2, label='Minor axis')
    
    plt.xlabel('U axis (m)')
    plt.ylabel('V axis (m)')
    plt.title(f"{title}\nMajor: {axes_2d[0]:.3f}m, Minor: {axes_2d[1]:.3f}m, Angle: {angle_2d:.1f}°")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()


def display_object_with_ellipsoid(pcd: o3d.geometry.PointCloud, ellipsoid_mesh: o3d.geometry.TriangleMesh, center_3d: np.ndarray = None, axes_3d: tuple[float, float, float] = None, window_name: str = "Object and Fitted Ellipsoid"):
    """
    Visualize 3D object point cloud with fitted ellipsoid mesh.
    
    Args:
        pcd: Original object point cloud
        ellipsoid_mesh: Fitted ellipsoid mesh
        center_3d: Optional ellipsoid center for coordinate frame display
        axes_3d: Optional axes dimensions for info display
        window_name: Visualization window name
    """
    # Color the point cloud
    pcd_colored = copy.deepcopy(pcd)
    pcd_colored.paint_uniform_color([0.2, 0.6, 0.9])  # Blue
    
    # Color the ellipsoid
    ellipsoid_colored = copy.deepcopy(ellipsoid_mesh)
    ellipsoid_colored.paint_uniform_color([0.9, 0.3, 0.3])  # Red
    ellipsoid_colored.compute_vertex_normals()
    
    geometries = [pcd_colored, ellipsoid_colored]
    
    # Add coordinate frame at center if provided
    if center_3d is not None:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=center_3d)
        geometries.append(coord_frame)
    
    # Create visualization window title with info
    if axes_3d is not None:
        title_info = f"{window_name} | Axes: [{axes_3d[0]:.3f}, {axes_3d[1]:.3f}, {axes_3d[2]:.3f}]m"
    else:
        title_info = window_name
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title_info,
        width=1200,
        height=900,
        mesh_show_back_face=True
    )


def visualize_viewpoints_around_object(
    pcd: o3d.geometry.PointCloud,
    ellipsoid_mesh: o3d.geometry.TriangleMesh,
    center_3d: np.ndarray,
    viewpoints: List[tuple[np.ndarray, np.ndarray]],
    num_viewpoints: int = 12,
    standoff_distance: float = 0.3,
    elevation_range: tuple[float, float] = (-90, 90),
    show_view_lines: bool = True,
    camera_frame_size: float = 0.05,
    window_name: str = "Object with Camera Viewpoints"):
    """
    Visualize object, ellipsoid, and camera viewpoints for robotic scanning.
    
    Args:
        pcd: Original object point cloud
        ellipsoid_mesh: Fitted ellipsoid mesh
        center_3d: (x, y, z) center of ellipsoid
        axes_3d: (semi_a, semi_b, semi_c) semi-axis lengths
        rotation_matrix: 3x3 rotation matrix for ellipsoid orientation
        num_viewpoints: number of viewpoints to generate
        standoff_distance: distance from ellipsoid surface (meters)
        elevation_range: (min_deg, max_deg) elevation angles for viewpoints
        show_view_lines: whether to draw lines from cameras to object center
        camera_frame_size: size of coordinate frames representing cameras
        window_name: visualization window name
    """    
    # Color the point cloud
    pcd_colored = copy.deepcopy(pcd)
    pcd_colored.paint_uniform_color([0.2, 0.6, 0.9])  # Blue
    
    # Color the ellipsoid (semi-transparent red)
    ellipsoid_colored = copy.deepcopy(ellipsoid_mesh)
    ellipsoid_colored.paint_uniform_color([0.9, 0.3, 0.3])  # Red
    ellipsoid_colored.compute_vertex_normals()
    
    geometries = [pcd_colored, ellipsoid_colored]
    
    # Add coordinate frame at object center
    center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=camera_frame_size * 2, origin=center_3d
    )
    geometries.append(center_frame)
    
    # Add camera coordinate frames at each viewpoint
    for i, (position, quaternion) in enumerate(viewpoints):
        # Create coordinate frame at camera position
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=camera_frame_size, origin=position
        )
        
        # Rotate the coordinate frame to match camera orientation
        # Convert quaternion to rotation matrix
        qx, qy, qz, qw = quaternion
        R_cam = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        
        camera_frame.rotate(R_cam, center=position)
        geometries.append(camera_frame)
        
        # Add sphere marker at camera position for better visibility
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=camera_frame_size * 0.3)
        sphere.translate(position)
        sphere.paint_uniform_color([0.1, 0.9, 0.1])  # Green
        geometries.append(sphere)
        
        # Add line from camera to object center
        if show_view_lines:
            line_points = np.array([position, center_3d])
            line_indices = [[0, 1]]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_points),
                lines=o3d.utility.Vector2iVector(line_indices)
            )
            line_set.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
            geometries.append(line_set)
    
    # Display information in title
    title_info = f"{window_name} | {num_viewpoints} viewpoints | Standoff: {standoff_distance:.2f}m"
    
    print(f"\n{'='*60}")
    print(f"VIEWPOINT VISUALIZATION")
    print(f"{'='*60}")
    print(f"Number of viewpoints: {num_viewpoints}")
    print(f"Standoff distance: {standoff_distance:.3f} m")
    print(f"Elevation range: {elevation_range[0]:.1f}° - {elevation_range[1]:.1f}°")
    print(f"Object center: ({center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f})")
    print(f"\nViewpoint positions (x, y, z):")
    for i, (pos, quat) in enumerate(viewpoints):
        print(f"  VP {i+1:2d}: ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")
    print(f"{'='*60}\n")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title_info,
        width=1400,
        height=1000,
        mesh_show_back_face=True
    )


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion (x, y, z, w).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        ndarray: quaternion [x, y, z, w]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:        
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    
    return np.array([x, y, z, w])


def ellipsoid_fitting_pipeline(pcd: o3d.geometry.PointCloud, plane_model: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray, o3d.geometry.TriangleMesh]:
    """
    Main pipeline to fit an ellipsoid to the object point cloud and visualize results.
    """
    # Step 1: Extract 2D perimeter from object projected onto table plane 
    perimeter_2d = extract_2d_convex_hull(pcd, plane_model)
    if debug:
        plot_2d_projection(pcd, plane_model, perimeter_2d)

    # Step 2: Fit 2D ellipse to the perimeter
    center_2d, axes_2d, angle_2d = fit_ellipse_to_2d_points(perimeter_2d)
    if debug:
        plot_ellipse_fit(perimeter_2d, center_2d, axes_2d, angle_2d)
    # Step 3: Transform 2D ellipse parameters to 3D ellipsoid
    height_margin = 0.01
    center_3d, axes_3d, rotation_matrix = convert_ellipse_to_3d(
        center_2d, axes_2d, angle_2d, plane_model, pcd, height_margin
    )

    # Step 4: Create the ellipsoid mesh
    ellipsoid_mesh = create_upper_ellipsoid_mesh(
        center_3d, axes_3d, rotation_matrix, resolution=20
    )
    if debug:
        display_object_with_ellipsoid(pcd, ellipsoid_mesh, center_3d, axes_3d)

    view_points = generate_camera_viewpoints_along_principal_axis(
        center_3d, axes_3d, rotation_matrix, plane_model, num_viewpoints=10, 
        standoff_distance=0.1)
    if debug:   
        visualize_viewpoints_around_object(pcd=pcd, ellipsoid_mesh=ellipsoid_mesh, viewpoints=view_points, center_3d=center_3d, num_viewpoints=12, standoff_distance=0.1,
                                            elevation_range=(-90, 90), show_view_lines=True, camera_frame_size=0.05)