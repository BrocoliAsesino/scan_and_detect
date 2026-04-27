#include "behaviortree_ros2/plugins.hpp"
// PCL Utils plugins
#include "plugins/action/compute_ellipsoid_fitting.hpp"
#include "plugins/action/filter_pcd.hpp"
#include "plugins/action/get_viewpoints.hpp"
#include "plugins/action/store_latest_pcd.hpp"

using namespace BT;

BT_REGISTER_ROS_NODES(factory, params) {
  // PCL Utils plugins
  factory.registerNodeType<StoreLatestPCD>("StoreLatestPCD", params);
  factory.registerNodeType<FilterPCD>("FilterPCD", params);
  factory.registerNodeType<ComputeEllipsoidFitting>("ComputeEllipsoidFitting",
                                                    params);
  factory.registerNodeType<GenerateViewPoints>("GenerateViewPoints", params);
};
