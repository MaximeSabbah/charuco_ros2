import os.path
from functools import partial


import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from sensor_msgs.msg import CameraInfo, Image

from message_filters import ApproximateTimeSynchronizer, Subscriber

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose

from realsense2_camera_msgs.msg import Extrinsics

from cv_bridge import CvBridge
from rclpy.time import Time, CONVERSION_CONSTANT

from sensor_msgs.msg import JointState
from std_msgs.msg import String

import pickle
import numpy as np
import pinocchio as pin

import cv2  # for image display and key handling

from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


class CalibrationDataCollector(Node):
    """ROS2 node that streams camera images and captures calibration data on key presses.

    This node subscribes to the RGB image and camera info topics, as well as the
    robot's joint states.  It maintains the latest image and camera info and
    displays the image continuously in an OpenCV window.  When the user presses
    the 's' key, the node captures the current image along with the camera
    intrinsic matrix, the transform from base to gripper, the transform from
    camera mount to camera, and the joint states.  The data are saved to
    sequentially numbered ``image_data_*.pkl`` files.  Pressing 'q' exits the
    loop and shuts down the node.
    """

    def __init__(self, data_directory: str = "data"):
        super().__init__("calibration_data_collector")

        # Ensure the output directory exists
        os.makedirs(data_directory, exist_ok=True)
        self._data_dir = data_directory
        # Determine the next file index
        i = 0
        while os.path.exists(os.path.join(self._data_dir, f"image_data_{i}.pkl")):
            i += 1
        self._file_index = i

        # Latest messages
        self.latest_image = None
        self.latest_camera_info = None
        self.latest_stamp = None
        self.latest_frame_id = None
        self._joint_state = None
        self._robot_description = None

        # Subscribers for image and camera info using approximate synchronisation
        filter_subscribers = [
            Subscriber(self, Image, "/camera/camera/color/image_raw"),
            Subscriber(self, CameraInfo, "/camera/camera/color/camera_info"),
        ]
        self._img_sync = ApproximateTimeSynchronizer(filter_subscribers, queue_size=10, slop=0.05)
        self._img_sync.registerCallback(self._image_callback)

        # Bridge for converting ROS images to OpenCV
        self._cvb = CvBridge()

        # Transform buffer and listener (non-spinning thread)
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self, spin_thread=True)

        # Subscribe to robot description (URDF) to know the kinematic chain is available
        self._robot_description_sub = self.create_subscription(
            String,
            "/robot_description",
            self._robot_description_cb,
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )

        # Subscribe to joint states
        self._joint_state_sub = self.create_subscription(
            JointState,
            "/franka/joint_states",
            self._joint_state_cb,
            20,
        )

    # Callbacks
    def _robot_description_cb(self, msg: String) -> None:
        self._robot_description = msg.data

    def _joint_state_cb(self, msg: JointState) -> None:
        self._joint_state = msg

    def _image_callback(self, image: Image, camera_info: CameraInfo) -> None:
        """Store the latest image and camera info for display and capture."""
        # Convert ROS image to OpenCV image
        encoding = "passthrough" if image.encoding == "rgb8" else "rgb8"
        self.latest_image = self._cvb.imgmsg_to_cv2(image, encoding)
        self.latest_camera_info = camera_info
        self.latest_stamp = image.header.stamp
        # Store the frame id of the image for transform lookup
        self.latest_frame_id = image.header.frame_id

    def save_current_frame(self) -> None:
        """Save the current frame and associated robot data to a pickle file."""
        if self.latest_image is None or self.latest_camera_info is None:
            self.get_logger().warning("No image available to save.")
            return
        if self._robot_description is None:
            self.get_logger().warning("Robot description not yet received; skipping save.")
            return
        if self._joint_state is None:
            self.get_logger().warning("Joint state not yet received; skipping save.")
            return
        # Look up transforms at the time of the latest image
        try:
            # base frame to gripper link (fer_link0 -> fer_link8)
            base_transform = self._buffer.lookup_transform(
                "fer_link0", "fer_link8", Time.from_msg(self.latest_stamp)
            )
        except Exception:
            self.get_logger().warning("Waiting for transform of base_transform")
            return
        try:
            # camera mount to camera optical frame
            camera_transform = self._buffer.lookup_transform(
                "camera_bottom_screw_frame",
                self.latest_frame_id,
                Time.from_msg(self.latest_stamp),
            )
        except Exception:
            self.get_logger().warning("Waiting for transform of camera_transform")
            return
        # Prepare joint map sorted to ensure consistent ordering
        joint_order = sorted(self._joint_state.name)
        joint_map = [self._joint_state.name.index(joint_name) for joint_name in joint_order]
        # Build data dictionary
        data = {
            "image_rgb": self.latest_image.copy(),
            "k_rgb": np.array(self.latest_camera_info.k),
            "base_transform": self.transform_msg_to_matrix(base_transform.transform),
            "camera_transform": self.transform_msg_to_matrix(camera_transform.transform),
            "joint_states": np.array(self._joint_state.position)[joint_map],
        }
        filename = os.path.join(self._data_dir, f"image_data_{self._file_index}.pkl")
        with open(filename, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.get_logger().info(f"Saved {filename}")
        self._file_index += 1

    # Helper to convert ROS transform message to 4×4 matrix
    def transform_msg_to_matrix(self, transform):
        return pin.XYZQUATToSE3(
            np.array(
                [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ]
            )
        ).np

    def transform_msg_to_matrix(self, transform):
        """Converts ROS Transform message into a 4x4 transformation matrix.

        :param transform: Transform message to convert into the matrix.
        :type transform: geometry_msgs.msg.Transform
        :return: Transformation matrix based on the ROS message.
        :rtype: Annotated[npt.NDArray[np.float64], Literal[4, 4]]
        """
        return pin.XYZQUATToSE3(
            np.array(
                [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ]
            )
        ).np


def main(args=None):
    """Entry point that runs the calibration data collector and handles UI."""
    rclpy.init(args=args)
    collector = CalibrationDataCollector(data_directory="data")
    window_name = "Calibration Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while rclpy.ok():
            # Process pending ROS callbacks
            rclpy.spin_once(collector, timeout_sec=0.1)
            # Display the latest image if available
            if collector.latest_image is not None:
                cv2.imshow(window_name, collector.latest_image)
            # Wait a short time for a key press; 1 ms to keep the UI responsive
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Save current frame
                collector.save_current_frame()
            elif key == ord('q'):
                # Quit the loop
                break
    finally:
        cv2.destroyAllWindows()
        collector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
