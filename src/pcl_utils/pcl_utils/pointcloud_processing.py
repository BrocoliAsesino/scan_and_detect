#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
import rclpy

from geometry_msgs.msg import Point
from shape_msgs.msg import Mesh, MeshTriangle
from mesh_msgs.msg import MeshGeometryStamped, MeshTriangleIndices


# Point Cloud Operations
def concatenate_point_clouds(tbl_pcl):
    for index, pcl in enumerate(tbl_pcl):
        if index == 0:
            main_pcl = pcl
        else:
            p_temp = np.concatenate((pcl.points, main_pcl.points))

            main_pcl.points = o3d.utility.Vector3dVector(p_temp)

    return main_pcl


def crop_point_cloud_with_polygon(pcl, json_path, visu):
    # crop
    crop_vol = o3d.visualization.read_selection_polygon_volume(json_path)
    cropped_pcl = crop_vol.crop_point_cloud(pcl)

    # crop visualisation
    if visu:
        line_set = visualize_selection_polygon_volume(crop_vol)
        cropped_pcl_colored = copy.deepcopy(cropped_pcl)
        cropped_pcl_colored.paint_uniform_color([1, 0.2, 0.2])
        o3d.visualization.draw_geometries([pcl, line_set, cropped_pcl_colored])

    return cropped_pcl


def remove_plane_from_point_cloud(
    pcl, distance_threshold=0.0013, ransac_n=3, num_iterations=600
):
    plane_model, inlier = pcl.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    outlier_cloud = pcl.select_by_index(inlier, invert=True)
    return outlier_cloud, plane_model


def apply_dbscan_clustering(
    pcl,
    eps=0.0008,
    min_points=10,
    print_progress=True,
    nbre_de_cluster_retour=1,
    seuil=0,
    type_return="tbl",
    visu_cluster=False,
):
    """
    Apply DBSCAN clustering to a point cloud.

    Args:
        pcl: Input point cloud
        eps: Maximum distance between two points to be considered neighbors
        min_points: Minimum number of points required to form a cluster
        print_progress: Whether to print progress
        nbre_de_cluster_retour: Number of largest clusters to return (0 = all clusters above threshold)
        seuil: Minimum cluster size threshold (used when nbre_de_cluster_retour=0)
        type_return: Return type - "tbl" (list of clusters), "pcl" (concatenated point cloud), or "labels" (numpy array of labels)
        visu_cluster: Whether to visualize clusters

    Returns:
        Depending on type_return:
        - "tbl": List of point clouds (one per cluster)
        - "pcl": Single concatenated point cloud
        - "labels": Numpy array of cluster labels for each point (-1 for noise)
    """
    if len(pcl.points) == 0:
        print("pcl vide en entre du dbscan cluster")
        if type_return == "labels":
            return np.array([])
        return pcl

    # Remove visualization in headless/non-interactive environments
    # o3d.visualization.draw_geometries([pcl])

    # creation des clusters et des tailles de cluster
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(pcl.cluster_dbscan(eps, min_points, print_progress))

    # If labels are requested, return them directly
    if type_return == "labels":
        return labels

    max_label = labels.max()

    # visualisation of cluster
    if visu_cluster:
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcl.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcl])

    # creation of a list of clusters
    cluster = []
    for i in range((max_label) + 1):
        tmp = np.where(labels == i)
        cluster.append(pcl.select_by_index(tmp[0]))

    # creation of an array of cluster sizes
    size = []
    for i in range(len(cluster)):
        size.append(len(cluster[i].points))

    print("Number of clusters: ", len(cluster))
    print("Cluster sizes: ", size)

    # return the n largest clusters
    if nbre_de_cluster_retour == 0:
        npsize = np.array(size)
        tmp = np.where(npsize > seuil)
        result = [cluster[i] for i in tmp[0]]

    else:
        # return the clusters by threshold on their size
        result = []
        for i in range(nbre_de_cluster_retour):
            result.append(cluster[size.index(max(size))])

    # format the result: as a list of clusters or as a single pcl
    if type_return == "tbl":
        pass

    elif type_return == "pcl":
        result = concatenate_point_clouds(result)

    return result


def remove_statistical_outliers(pcl):
    nb_neighbours = 20
    std_ratio = 2.0
    cl, ind = pcl.remove_statistical_outlier(nb_neighbours, std_ratio)
    # visualize_inliers_and_outliers(cl, ind)  # Disabled visualization for headless environments
    return cl  # cl is already the filtered inlier cloud


# Mesh Operations
def clean_triangle_mesh(mesh):
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles()
    )
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    # mesh_clean = copy.deepcopy(bpa_mesh)

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < max(
        cluster_n_triangles
    )
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def poisson_reconstruction(pcl):
    # get bb du pcl pour crop le stl
    bbox = pcl.get_oriented_bounding_box()

    # calculate and re-orient normals
    pcl.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcl.orient_normals_to_align_with_direction(orientation_reference=[0.0, 0.0, 1.0])
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcl, depth=8
    )
    densities = np.array(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # crop the mesh
    mesh = mesh.crop(bbox)

    # clean the mesh (useless with poisson?)
    mesh = clean_triangle_mesh(mesh)

    mesh = mesh.filter_smooth_simple(number_of_iterations=2)
    mesh.compute_vertex_normals()
    print("smooth done")

    return mesh


def ball_pivoting_mesh_reconstruction(pcl):
    # pcl = apply_dbscan_clustering(pcl, eps=0.0006, min_points=10, print_progress=_debug, nbre_de_cluster_retour=1, seuil=0, type_return="pcl", visu_cluster=_debug)

    pcl = pcl.voxel_down_sample(voxel_size=0.001)

    pcl.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    print("normal calculated")
    pcl.orient_normals_to_align_with_direction(orientation_reference=[0.0, 0.0, 1.0])
    print("normal oriented")
    distances = pcl.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3.2 * avg_dist

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcl, o3d.utility.DoubleVector([radius, radius * 2])
    )
    print("bpa done")
    # clean the mesh

    print("clean done")
    mesh = bpa_mesh.filter_smooth_simple(number_of_iterations=2)
    mesh = clean_triangle_mesh(mesh)
    mesh.compute_vertex_normals()
    print("smooth done")
    return mesh


def convert_mesh_to_ros_geometry_msg(mesh):
    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.triangles)

    message = MeshGeometryStamped()
    message.uuid = "msg"
    message.header.stamp = rclpy.time.Time().to_msg()
    message.header.frame_id = "world"

    for i in range(len(vertices)):
        Message_point = Point()
        Message_point.x = vertices[i][0]
        Message_point.y = vertices[i][1]
        Message_point.z = vertices[i][2]
        message.mesh_geometry.vertices.append(Message_point)

    for i in range(len(triangles)):
        Message_triangles = MeshTriangleIndices()
        Message_triangles.vertex_indices = triangles[i]
        message.mesh_geometry.faces.append(Message_triangles)

    return message


def convert_mesh_to_ros_msg(mesh):
    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.triangles)
    message = Mesh()

    for i in range(len(vertices)):
        Message_point = Point()
        Message_point.x = vertices[i][0]
        Message_point.y = vertices[i][1]
        Message_point.z = vertices[i][2]
        message.vertices.append(Message_point)

    for i in range(len(triangles)):
        Message_triangles = MeshTriangle()
        Message_triangles.vertex_indices = triangles[i]
        message.triangles.append(Message_triangles)

    return message


# Registration
def perform_icp_registration(source, target, ICP_type="PointToPoint"):
    if not target.has_points():
        return source

    source = copy.deepcopy(source)
    target = copy.deepcopy(target)

    source.voxel_down_sample(voxel_size=0.0001)
    target.voxel_down_sample(voxel_size=0.0001)

    source_cordon = remove_plane_from_point_cloud(source)
    o3d.visualization.draw_geometries([source_cordon])

    target_cordon = remove_plane_from_point_cloud(target)
    o3d.visualization.draw_geometries([target_cordon])

    filtred_source_cordon = apply_dbscan_clustering(
        source_cordon,
        eps=0.0008,
        min_points=10,
        print_progress=True,
        nbre_de_cluster_retour=0,
        seuil=800,
        type_return="pcl",
    )
    o3d.visualization.draw_geometries([filtred_source_cordon])

    filtred_target_cordon = apply_dbscan_clustering(
        target_cordon,
        eps=0.0008,
        min_points=10,
        print_progress=True,
        nbre_de_cluster_retour=0,
        seuil=800,
        type_return="pcl",
    )
    o3d.visualization.draw_geometries([filtred_target_cordon])

    crit = o3d.pipelines.registration.ICPConvergenceCriteria()
    crit.relative_rmse = 0.0000001
    crit.max_iteration = 350
    crit.relative_fitness = 0.00000000000000001
    threshold = 1
    trans_init = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    print("initial alignement")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init
    )
    print(evaluation)

    if ICP_type == "PointToPoint":
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            crit,
        )

    if ICP_type == "PointToPlane":
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50)
        )
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50)
        )
        rclpy.logging.get_logger("ICP").info("normal estimation for ICP done")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            crit,
        )

    print(reg_p2p)
    print(reg_p2p.transformation)
    # final transform from ICP
    source.transform(reg_p2p.transformation)

    return source


# Visualization
def visualize_selection_polygon_volume(polyG):
    # Only works with polygones of type cube
    points = polyG.bounding_polygon
    axis = polyG.orthogonal_axis
    axis_min = polyG.axis_min
    axis_max = polyG.axis_max
    points_haut = []
    points_bas = []
    for point in points:
        if axis == "Z":
            tmp_ = copy.deepcopy(point)
            tmp_[2] = tmp_[2] + axis_max
            points_haut.append(tmp_)

            tmp_ = copy.deepcopy(point)
            tmp_[2] = tmp_[2] + axis_min
            points_bas.append(tmp_)

    points = points_haut + points_bas
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return line_set


def visualize_inliers_and_outliers(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


# Mesh Operations
def generate_upper_ellipsoid_mesh(
    center_3d: np.ndarray,
    axes_3d: tuple[float, float, float],
    rotation_matrix: np.ndarray,
    resolution: int = 20,
) -> o3d.geometry.TriangleMesh:
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
    # upper_mask = vertices[:, 2] >= 0

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
