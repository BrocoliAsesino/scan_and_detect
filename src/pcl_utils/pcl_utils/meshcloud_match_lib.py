import open3d as o3d
import os
import numpy as np
import copy



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Preprocessing pointclouds!", flush=True)
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(cad_pcl_points, perception_pcl_points, voxel_size, initial_transform):
    print(":: Preparing dataset... here are the initial poses of your pointclouds", flush=True)

    source = cad_pcl_points
    target = perception_pcl_points

    source.transform(initial_transform)
    o3d.visualization.draw_geometries([source, target])

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: executing RANSAC registration on downsampled point clouds.", flush=True)
    #print("   Since the downsampling voxel size is %.3f," % voxel_size, flush=True)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold, flush=True)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    print(":: executing ICP registration", flush=True)
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    distance_threshold = voxel_size * 0.5

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation, o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result


# -----------------------------------------------------------------------


def registration(pcl_cad, pcl_perception, initial_transform, debug_display=True, global_method='choose your fighter'):
    voxel_size = 0.02
    pcl_cad_copy = copy.deepcopy(pcl_cad)
    pcl_perception_copy = copy.deepcopy(pcl_perception)

    if global_method == 'convex_hull':
        print("Chosen registration method: ", global_method, flush=True)
        # Compute convex hulls of the point clouds
        cad_ch_mesh, _ = pcl_cad.compute_convex_hull()
        perception_ch_mesh, _ = pcl_perception.compute_convex_hull()
        cad_ch_mesh_points = cad_ch_mesh.sample_points_uniformly(number_of_points=22000)
        perception_ch_mesh_points = perception_ch_mesh.sample_points_uniformly(number_of_points=22000)
        o3d.visualization.draw_geometries([pcl_cad, pcl_perception])

        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(cad_ch_mesh_points, perception_ch_mesh_points, voxel_size, initial_transform)

        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        print("\nThe first transformation is:\n", result_ransac.transformation, flush=True)
        if debug_display:
            draw_registration_result(source, target, result_ransac.transformation)

        ## NEW 
        # print("\n############ ABOUT TO APPLY A SECOND RANSAC WITHOUT CH ############:\n", flush=True)
        # new_voxel_size = 0.02
        # source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pcl_cad, pcl_perception, new_voxel_size, result_ransac.transformation)
        # print("\n############ EL NUEVO SOURCE Y TARGET ############:\n", flush=True)
        # identity = np.eye(4)
        # draw_registration_result(source, target, identity)
        # result_ransac2 = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        # print("\nThe INTERMEDIATE transformation is:\n", result_ransac2.transformation, flush=True)
        # if debug_display:
        #     draw_registration_result(source, target, result_ransac2.transformation)

        print("\n############ ABOUT TO PERFORM ICP ############:\n", flush=True)
        voxel_size = 0.005  # Changed from 0.02 to 0.005 for ICP
        # total_transformation = result_ransac2.transformation @ result_ransac.transformation
        # result_ransac2.transformation = total_transformation
        # print("\nThe TOTAL transformation is:\n", total_transformation, flush=True)
        result_icp = refine_registration(pcl_cad, pcl_perception, None, None, voxel_size, result_ransac)
        print("\nThe resulting transformation is:\n", result_icp.transformation, flush=True)
        if debug_display:
            draw_registration_result(pcl_cad_copy, pcl_perception_copy, result_icp.transformation)



    elif global_method == 'RANSAC':
        print("Chosen registration method: ", global_method, flush=True)
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pcl_cad, pcl_perception, voxel_size, initial_transform)

        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        if debug_display:
            draw_registration_result(source, target, result_ransac.transformation)

        # Refine registration using ICP
        result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac)
        if debug_display:
            draw_registration_result(source, target, result_icp.transformation)

    else:
        # Handle invalid method name
        print("Invalid global registration method: ", global_method, flush=True)
        return np.eye(4)

    return result_icp.transformation, result_icp.fitness




if __name__ == "__main__":
    path_pcdCAO = os.getcwd() + "/originalCAO/pcl_from_CAO.ply"

    # pcdCAO = o3d.io.read_point_cloud(pathCAO)

    path_pcdPerception = os.getcwd() + "/results/23/result.ply"
    # pcdPerception = o3d.io.read_point_cloud(pathPerception)

    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    # o3d.visualization.draw_geometries([pcdCAO, pcdPerception, mesh])

    # trG, trICP = registration(pcdCAO, pcdPerception)

    path_original_CAO = os.getcwd() + "/originalCAO/surface_process_from_CAO.stp"
    path_result = os.getcwd() + "/results/surface_recallee.stp"
    recal_and_move(path_pcdCAO, path_pcdPerception, path_original_CAO, path_result, True)
    # move_CAO(path_to_surface_from_CAO, path_result, trICP)

    # *****************DEBUG CODE ( visualisation)***********************
    # import copy

    # pcdCAO_copy = copy.deepcopy(pcdCAO)
    # pcdCAO_copy.transform(trICP)
    # pcdCAO_copy.paint_uniform_color([1, 0, 0])

    # pcdCAO.paint_uniform_color([0, 1, 0])
    # pcdPerception.paint_uniform_color([0, 0, 1])

    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    # o3d.visualization.draw_geometries([pcdCAO, pcdCAO_copy, pcdPerception, mesh])
    # #  o3d.visualization.draw_geometries([pcdCAO_copy, pcdPerception, mesh])

    # tmp = os.getcwd() + "/results/perception_recalee.ply"
    # o3d.io.write_point_cloud(tmp, pcdCAO_copy)
    # # first rough alignement with box center
    # # pcdCAO_box = pcdCAO.get_oriented_bounding_box()
    # # pcdCAO_center = pcdCAO_box.get_center()

    # # print(pcdCAO_center)
    # # pcdPerception_box = pcdPerception.get_oriented_bounding_box()
    # # pcdPerception_center = pcdPerception_box.get_center()

    # # translation = pcdPerception_center - pcdCAO_center
    # # pcdCAO_boxrecal = copy.deepcopy(pcdCAO).translate(translation)
