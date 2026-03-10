# EUT YOLO Interfaces

This package (`eut_yolo_interfaces`) defines the custom ROS 2 messages and services used for communication between the different nodes within the `eut_yolo` perception system. Defining interfaces in a dedicated package promotes modularity and reusability.

## Overview

The `eut_yolo` system relies on various nodes performing tasks like object detection, segmentation, tracking, and pose estimation. These nodes need to exchange data efficiently and reliably. This package provides the standardized data structures (messages) and communication contracts (services) for this purpose.

## Defined Interfaces

The interfaces are organized into `msg/` and `srv/` subdirectories.

### Messages (`msg/`)

1.  **`Perception.msg`**: Represents comprehensive data for a single perceived object instance, potentially tracked over time.
    *   `uint32 track_id`: Unique identifier for the tracked object.
    *   `string label`: Class label of the detected object (e.g., "box", "person").
    *   `float32 confidence`: Confidence score of the detection (typically 0.0 to 1.0).
    *   `uint32 class_id`: Numerical class ID.
    *   `bool detection_available`: Flag indicating if bounding box detection is available.
    *   `bool segmentation_available`: Flag indicating if segmentation mask info is available.
    *   `bool depth_available`: Flag indicating if depth information is available.
    *   `bool pose_available`: Flag indicating if pose estimation is available.
    *   `uint32 max_history_length`: Maximum buffer size for historical data (e.g., poses).
    *   `uint32 num_detection_samples`: Number of detection samples currently stored/averaged.
    *   `uint32 num_segmentation_samples`: Number of segmentation samples stored/averaged.
    *   `uint32 num_depth_samples`: Number of depth samples stored/averaged.
    *   `uint32 num_pose_samples`: Number of pose samples stored/averaged.
    *   `uint32[] bbox_xyxy`: Bounding box coordinates [x_min, y_min, x_max, y_max] in pixel space.
    *   `float32[] bbox_centroid`: Centroid of the bounding box [x, y] in pixel space.
    *   `float32[] mask_centroid`: Centroid of the segmentation mask [x, y] in pixel space.
    *   `float32[] resize_ratio`: Ratio used for resizing image during inference [ratio_x, ratio_y].
    *   `float32 average_r_color_channel`: Average Red color channel value within the mask/bbox.
    *   `float32 average_g_color_channel`: Average Green color channel value.
    *   `float32 average_b_color_channel`: Average Blue color channel value.
    *   `float32 depth`: Estimated depth of the object (e.g., distance from camera in meters).
    *   `float32 valid_depth_mask_percentage`: Percentage of valid depth pixels within the mask.
    *   `geometry_msgs/TransformStamped pose`: Estimated 6D pose of the object relative to a frame_id defined in the header.
    *   `float32 bbox_width_m`: Width of the bounding box in meters (requires depth/camera info).
    *   `float32 bbox_height_m`: Height of the bounding box in meters.
    *   `float32 mask_dim1_m`: Primary dimension of the mask in meters.
    *   `float32 mask_dim2_m`: Secondary dimension of the mask in meters.
    *   `float32 std_dev_*`: Standard deviations for various estimated values (confidence, centroids, depth, pose, dimensions), indicating uncertainty.

2.  **`PerceptionObjects.msg`**: Represents a collection of perceived objects detected in a single frame or time step.
    *   `Perception[] perception_objects`: An array/list containing multiple `Perception` messages, one for each object detected.

### Services (`srv/`)

1.  **`PerformInference.srv`**: Defines a service to request an on-demand perception inference run.
    *   **Request:**
        *   `string prompt`: A textual prompt to guide the inference (e.g., specific object class, attribute).
        *   `int32 num_poses_to_average`: Number of historical poses to average for the result.
        *   `float32 timeout_sec`: Maximum time (in seconds) to wait for the service response.
    *   **Response:**
        *   `Perception[] perception_objects`: An array/list of `Perception` messages for the objects detected based on the request.
        *   `bool success`: Flag indicating if the inference service call was successful.
        *   `string message`: An optional status or error message.

## Usage

### Building

This package is a standard ROS 2 package. To use the defined interfaces, ensure this package is built as part of your Colcon workspace:

```bash
# Navigate to your workspace root
cd ~/workspace # Or your specific workspace path

# Build the workspace (including eut_yolo_interfaces)
colcon build --packages-select eut_yolo_interfaces
# Or build the entire workspace
colcon build
```

### Dependencies

Other ROS 2 packages within the `eut_yolo` system (or external packages interacting with it) that need to use these custom messages or services must declare a dependency on `eut_yolo_interfaces` in their `package.xml`:

```xml
<depend>eut_yolo_interfaces</depend>
```

And in their `CMakeLists.txt`:

```cmake
find_package(eut_yolo_interfaces REQUIRED)

# Link against the interfaces library if needed (e.g., for C++)
ament_target_dependencies(your_node_or_library eut_yolo_interfaces)
```

### Using Interfaces

Once the package is built and dependencies are correctly set up, you can import and use the messages and services in your C++ or Python nodes just like standard ROS 2 interfaces.

**Example (Python):**

```python
# Import custom messages
from eut_yolo_interfaces.msg import Perception, PerceptionObjects

# Import a custom service
from eut_yolo_interfaces.srv import PerformInference

# ... rest of your node code ...

# Example: Creating a subscriber
# self.create_subscription(PerceptionObjects, '/perception/objects', self.perception_callback, 10)

# Example: Creating a service client
# self.inference_client = self.create_client(PerformInference, '/perception/perform_inference')
# request = PerformInference.Request()
# request.prompt = "detect the blue cube"
# request.num_poses_to_average = 5
# request.timeout_sec = 10.0
# future = self.inference_client.call_async(request)

```

**Example (C++):**

```cpp
// Include custom messages
#include <eut_yolo_interfaces/msg/perception.hpp>
#include <eut_yolo_interfaces/msg/perception_objects.hpp>

// Include a custom service
#include <eut_yolo_interfaces/srv/perform_inference.hpp>

// ... rest of your node code ...

// Example: Creating a subscriber
// using PerceptionObjects = eut_yolo_interfaces::msg::PerceptionObjects;
// subscription_ = this->create_subscription<PerceptionObjects>(
//   "/perception/objects", 10, std::bind(&MyNode::perception_callback, this, std::placeholders::_1));

// Example: Creating a service client
// using PerformInference = eut_yolo_interfaces::srv::PerformInference;
// client_ = this->create_client<PerformInference>("/perception/perform_inference");
// auto request = std::make_shared<PerformInference::Request>();
// request->prompt = "detect the blue cube";
// request->num_poses_to_average = 5;
// request->timeout_sec = 10.0;
// auto future_result = client_->async_send_request(request);

```

## Contribution

If you need to add new message or service types for the `eut_yolo` system, please add the corresponding `.msg` or `.srv` files to the respective directories in this package and update the `CMakeLists.txt` accordingly. Ensure that the interface design is clear, efficient, and well-documented.
