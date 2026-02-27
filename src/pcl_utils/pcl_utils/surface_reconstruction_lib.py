#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
import rclpy

from geometry_msgs.msg import PointStamped
from shape_msgs.msg import Mesh, MeshTriangle
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from rclpy.node import Node
from rclpy.qos import QoSProfile


def concat_PCL(tbl_pcl):
    for index, pcl in enumerate(tbl_pcl):
        if index == 0:
            main_pcl = pcl
        else:
            p_temp = np.concatenate((pcl.points, main_pcl.points))

            main_pcl.points = o3d.utility.Vector3dVector(p_temp)

    return main_pcl


def crop_with_polygon(pcl, json_path, visu):
    # crop
    crop_vol = o3d.visualization.read_selection_polygon_volume(json_path)
    cropped_pcl = crop_vol.crop_point_cloud(pcl)

    # crop visualisation
    if visu:
        line_set = display_selection_polygon_volume(crop_vol)
        cropped_pcl_colored = copy.deepcopy(cropped_pcl)
        cropped_pcl_colored.paint_uniform_color([1, 0.2, 0.2])
        o3d.visualization.draw_geometries([pcl, line_set, cropped_pcl_colored])

    return cropped_pcl


def display_selection_polygon_volume(polyG):
    # ATTENTION, CETTE FONCTION NE FONCTIONNE QUE POUR DES POLYGONES DE TYPE CUBE...
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
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return line_set


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def clean_mesh(mesh):
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    # mesh_clean = copy.deepcopy(bpa_mesh)

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < max(cluster_n_triangles)
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
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcl.orient_normals_to_align_with_direction(orientation_reference=[0.0, 0.0, 1.0])
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl, depth=8)
    densities = np.array(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # crop the mesh
    mesh = mesh.crop(bbox)

    # clean the mesh (useless with poisson?)
    mesh = clean_mesh(mesh)

    mesh = mesh.filter_smooth_simple(number_of_iterations=2)
    mesh.compute_vertex_normals()
    print("smooth done")

    return mesh


def ball_pivoting(pcl):

    # pcl = db_scan_filter(pcl, eps=0.0006, min_points=10, print_progress=_debug, nbre_de_cluster_retour=1, seuil=0, type_return="pcl", visu_cluster=_debug)

    pcl = pcl.voxel_down_sample(voxel_size=0.001)

    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    print("normal calculated")
    pcl.orient_normals_to_align_with_direction(orientation_reference=[0.0, 0.0, 1.0])
    print("normal oriented")
    distances = pcl.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3.2 * avg_dist

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcl, o3d.utility.DoubleVector([radius, radius * 2]))
    print("bpa done")
    # clean the mesh

    print("clean done")
    mesh = bpa_mesh.filter_smooth_simple(number_of_iterations=2)
    mesh = clean_mesh(mesh)
    mesh.compute_vertex_normals()
    print("smooth done")
    return mesh


def supress_plane(pcl):
    plane_model, inlier = pcl.segment_plane(distance_threshold=0.0013, ransac_n=3, num_iterations=600)
    outlier_cloud = pcl.select_by_index(inlier, invert=True)
    return outlier_cloud


def supress_plane(pcl):
    plane_model, inlier = pcl.segment_plane(distance_threshold=0.0013, ransac_n=3, num_iterations=600)
    outlier_cloud = pcl.select_by_index(inlier, invert=True)
    return outlier_cloud


def db_scan_filter(pcl, eps=0.0008, min_points=10, print_progress=True, nbre_de_cluster_retour=1, seuil=0, type_return="tbl", visu_cluster=False):

    if len(pcl.points) == 0:
        print("pcl vide en entre du dbscan cluster")
        return pcl
    
    o3d.visualization.draw_geometries([pcl])

    # creation des clusters et des tailles de cluster
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcl.cluster_dbscan(eps, min_points, print_progress))

    max_label = labels.max()

    # visulisation des cluster
    if visu_cluster == True:
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcl.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcl])

    # creation d'un tableau des cluster
    cluster = []
    for i in range((max_label) + 1):
        tmp = np.where(labels == i)
        cluster.append(pcl.select_by_index(tmp[0]))

    # creation d'un tableau des tailles de cluster
    size = []
    for i in range(len(cluster)):
        size.append(len(cluster[i].points))

    print("Number of clusters: ", len(cluster))
    print("Cluster sizes: ", size)

    # retour des n plus grands cluster
    if nbre_de_cluster_retour == 0:
        npsize = np.array(size)
        tmp = np.where(npsize > seuil)
        result = [cluster[i] for i in tmp[0]]

    else:
        # retour des clusters par threshold sur leur taille
        result = []
        for i in range(nbre_de_cluster_retour):
            result.append(cluster[size.index(max(size))])

    # formatage du resultat : en tableau de custer ou en un seul pcl
    if type_return == "tbl":
        pass

    elif type_return == "pcl":
        result = concat_PCL(result)

    return result


def ICP_cordon(source, target, ICP_type="PointToPoint"):
    if not target.has_points():
        return source

    source = copy.deepcopy(source)
    target = copy.deepcopy(target)

    source.voxel_down_sample(voxel_size=0.0001)
    target.voxel_down_sample(voxel_size=0.0001)

    source_cordon = supress_plane(source)
    o3d.visualization.draw_geometries([source_cordon])

    target_cordon = supress_plane(target)
    o3d.visualization.draw_geometries([target_cordon])

    filtred_source_cordon = db_scan_filter(source_cordon, eps=0.0008, min_points=10, print_progress=True, nbre_de_cluster_retour=0, seuil=800, type_return="pcl")
    o3d.visualization.draw_geometries([filtred_source_cordon])

    filtred_target_cordon = db_scan_filter(target_cordon, eps=0.0008, min_points=10, print_progress=True, nbre_de_cluster_retour=0, seuil=800, type_return="pcl")
    o3d.visualization.draw_geometries([filtred_target_cordon])

    crit = o3d.pipelines.registration.ICPConvergenceCriteria()
    crit.relative_rmse = 0.0000001
    crit.max_iteration = 350
    crit.relative_fitness = 0.00000000000000001
    threshold = 1
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    print("initial alignement")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    if ICP_type == "PointToPoint":
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(), crit)

    if ICP_type == "PointToPlane":
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
        rospy.loginfo("normal estimation for ICP done")
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane(), crit)

    print(reg_p2p)
    print(reg_p2p.transformation)
    # final transform from ICP
    source.transform(reg_p2p.transformation)

    return source


def open_3d_mesh_to_ros_GeometryMeshMsg(mesh):

    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.triangles)

    message = MeshGeometryStamped()
    message.uuid="msg"
    message.header.stamp=rospy.get_rostime()
    message.header.frame_id="world"

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

def open_3d_mesh_to_ros_msg(mesh):

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


def statistical_outlier_removal(pcl):
    nb_neighbours = 20
    std_ratio = 2.0
    cl, ind = pcl.remove_statistical_outlier(nb_neighbours, std_ratio)
    # display_inlier_outlier(cl, ind)  # Disabled visualization for headless environments
    return cl  # cl is already the filtered inlier cloud


import time
import open3d as o3d
import numpy as np
from math import ceil
from statistics import mean
import copy

# import sys
# sys.path.append('/home/agonzalezhernandez/ros2_ws/src/pcl_concatenator/')
# from OCC.Core.AIS import AIS_Shape

# from OCC.Core.gp import gp_Pnt
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
# from OCC.Core.TColgp import TColgp_Array2OfPnt
# from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
# from OCC.Core.GeomAbs import GeomAbs_C2, GeomAbs_C3
# from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface, shapeanalysis
# from OCC.Display.SimpleGui import init_display
# # from OCC.Extend.DataExchange import read_step_file



# def nurbs_from_pcl(pcd0):
#     # display, start_display, add_menu, add_function_to_menu = init_display()

#     # pcd0=statistical_outlier_removal(pcdbrut)

#     voxel_size = 0.004

#     # pcd0=db_scan_filter(pcd0, eps=0.0008, min_points=1, print_progress=True, nbre_de_cluster_retour=1, seuil=0, type_return="pcl", visu_cluster=True)

#     # pcd = o3d.io.read_point_cloud("final_pcl3mm_rotated.ply")

#     ## GET BB

#     # bb1=o3d.geometry.OrientedBoundingBox()
#     # bb1=bb1.create_from_points(pcd0,True)

#     bb1 = pcd0.get_oriented_bounding_box()
#     bb_dim = bb1.extent
#     bb_rot = bb1.R
#     cropxy = 0.003
#     bb = o3d.geometry.OrientedBoundingBox(center=bb1.center, R=bb_rot, extent=np.array([bb_dim[0] - cropxy, bb_dim[1] - cropxy, bb_dim[2]]))
#     pcd = pcd0.crop(bb)
#     bb_dim = bb.extent
#     # o3d.visualization.draw_geometries([pcd, bb])
#     # o3d.visualization.draw_geometries([pcd0, bb,bb1])

#     bb_points = bb.get_box_points()

#     Lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(bb)

#     ## on trouve tout les points connectés au points 0
#     point_to_keep = [np.asarray(bb_points)[0]]
#     for couple_points in np.asarray(Lines.lines).tolist():
#         if 0 in couple_points:
#             couple_points.remove(0)
#             point_to_keep.append(np.asarray(bb_points)[couple_points[0]])

#     ## On créer les trois vecteurs du repère de la bounding box
#     vector = []
#     for i in range(1, len(point_to_keep)):
#         vector.append(point_to_keep[i] - point_to_keep[0])

#     ##On retire le plus petit vecteur (Z), on s'en fout ici
#     longueur = []
#     for v in vector:
#         longueur.append(np.linalg.norm(v))
#     min_longueur = min(longueur)
#     min_index = longueur.index(min_longueur)
#     vector.pop(min_index)

#     ## on défini le nombre de voxel suivant les vecteurs
#     nbX = ceil(np.linalg.norm(vector[0]) / voxel_size)
#     nbY = ceil(np.linalg.norm(vector[1]) / voxel_size)

#     # on rends les vecteurs unitaires:
#     unit_vector = []
#     for v in vector:
#         new_v = v / np.linalg.norm(v)
#         unit_vector.append(new_v)

#     array = TColgp_Array2OfPnt(1, nbX - 1, 1, nbY - 1)

#     BOX = []
#     points = []
#     for j in range(0, nbY - 1):
#         for i in range(0, nbX - 1):

#             transf1 = unit_vector[0] * i * voxel_size
#             transf2 = unit_vector[1] * j * voxel_size

#             new_point = []
#             for c in range(len(transf1)):
#                 new_point.append(point_to_keep[0][c] + transf1[c] + transf2[c])
#             box = o3d.geometry.OrientedBoundingBox(center=np.array(new_point), R=bb_rot, extent=np.array([voxel_size, voxel_size, 1]))
#             BOX.append(box)
#             tmp = pcd.crop(box)
#             # si il y a des point dans le voxel
#             if tmp.points:
#                 listeZ = [p[2] for p in (tmp.points)]
#                 averageZ = mean(listeZ)
#             # si il n'y a pas de points dans le voxel
#             else:
#                 # s'il y a une valeur precente on la prends
#                 if "averageZ" in locals():
#                     pass
#                 # sinon on agrandi le voxel jusqu'a trouver au moins un point
#                 else:
#                     voxel_scaling_factor = 1
#                     while not tmp.points:
#                         print("agrandissement du voxel")
#                         box = o3d.geometry.OrientedBoundingBox(
#                             center=np.array(new_point), R=bb_rot, extent=np.array([voxel_size * voxel_scaling_factor, voxel_size * voxel_scaling_factor, 1])
#                         )
#                         tmp = pcd.crop(box)
#                         voxel_scaling_factor = voxel_scaling_factor + 0.5
#                     listeZ = [p[2] for p in (tmp.points)]
#                     averageZ = mean(listeZ)
#                     print(averageZ)
#                     print(type(averageZ))
#                     # averageZ = bb.get_max_bound()[2]

#             occ_point = gp_Pnt(new_point[0], new_point[1], averageZ).Scaled(gp_Pnt(), 1000)

#             points.append(occ_point)
#             # if i < 3:
#             #     display.DisplayShape(occ_point, color="GREEN")
#             # else:
#             #     display.DisplayShape(occ_point, color="GREEN")

#             array.SetValue(i + 1, j + 1, occ_point)
#     # open3d_display_element = [pcd]
#     # open3d_display_element.append(BOX)
#     # o3d.visualization.draw_geometries(open3d_display_element)

#     bspl_surf = GeomAPI_PointsToBSplineSurface(array, 2, 2, GeomAbs_C2, 0.0001).Surface()

#     return bspl_surf
#     # face = BRepBuilderAPI_MakeFace(bspl_surf, 1e-4).Face()
#     # display.DisplayShape(face, color="BLUE")

#     # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

#     # from OCC.Extend.DataExchange import read_stl_file
#     # stl_filename = "final_mesh.stl"
#     # stl_shp = read_stl_file(stl_filename)
#     # display.DisplayShape(stl_shp,color="GREEN")

#     # start_display()


# # def visualize_step_file(file_paths):
# #     display, start_display, add_menu, add_function_to_menu = init_display()
# #     ais_context = display.GetContext()
# #     dc = ais_context.DeviationCoefficient()
# #     da = ais_context.DeviationAngle()
# #     print(dc)
# #     print(da)
# #     factor = 5
# #     ais_context.SetDeviationCoefficient(dc / factor)
# #     ais_context.SetDeviationAngle(da / factor)

# #     color = ["BLUE", "RED", "GREEN"]
# #     for index, path in enumerate(file_paths):
# #         CAO = read_step_file(path, as_compound=True, verbosity=True)
# #         display.DisplayShape(CAO, color=color[index])

# #     start_display()

# import ifcopenshell

# def visualize_step_file(file_paths):
#     display, start_display, add_menu, add_function_to_menu = init_display()
#     ais_context = display.GetContext()
#     dc = ais_context.DeviationCoefficient()
#     da = ais_context.DeviationAngle()
#     print(dc)
#     print(da)
#     factor = 5
#     ais_context.SetDeviationCoefficient(dc / factor)
#     ais_context.SetDeviationAngle(da / factor)

#     color = ["BLUE", "RED", "GREEN"]
#     for index, path in enumerate(file_paths):
#         model = ifcopenshell.open(path)
#         shape = model.geometry
#         display.DisplayShape(shape, color=color[index])

#     start_display()
