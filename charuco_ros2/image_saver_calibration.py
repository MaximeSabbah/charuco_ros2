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

from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__("minimal_publisher")

        i = 0
        self._filename = f"data/image_data_{i}.pkl"
        while os.path.exists(self._filename):
            i += 1
            self._filename = f"data/image_data_{i}.pkl"

        filter_subscribers = [
            Subscriber(self, Image, "/camera/camera/color/image_raw"),
            Subscriber(self, CameraInfo, "/camera/camera/color/camera_info"),
        ]

        image_approx_time_sync = ApproximateTimeSynchronizer(
            filter_subscribers, queue_size=1000, slop=0.01
        )
        # Register callback depending on the configuration
        image_approx_time_sync.registerCallback(self._on_image_data_cb)

        # Image type converter
        self._cvb = CvBridge()

        # Transform buffers
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self, spin_thread=True)

        self._robot_description = None
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

        self._joint_state = None
        self._joint_state_sub = self.create_subscription(
            JointState, "/franka/joint_states", self._joint_state_cb, 20
        )
        print("Started")

    def _robot_description_cb(self, msg: String) -> None:
        self._robot_description = msg.data

    def _joint_state_cb(self, msg: JointState) -> None:
        self._joint_state = msg

    def _on_image_data_cb(self, image: Image, camera_info: CameraInfo):
        if self._robot_description is None:
            print("Waiting for robot description")
            return

        if self._joint_state is None:
            print("Waiting for joint states")
            return

        encoding = "passthrough" if image.encoding == "rgb8" else "rgb8"
        image_rgb = self._cvb.imgmsg_to_cv2(image, encoding)

        try:
            base_transform = self._buffer.lookup_transform(
                "fer_link0", "fer_link8", Time.from_msg(image.header.stamp)
            )
        except:
            print("waiting for transform of base_transform")
            return

        try:
            camera_transform = self._buffer.lookup_transform(
                "camera_bottom_screw_frame",
                image.header.frame_id,
                Time.from_msg(image.header.stamp),
            )
        except:
            print("waiting for transform of camera_transform")
            return

        joint_order = sorted(self._joint_state.name)
        joint_map = [self._joint_state.name.index(joint_name) for joint_name in joint_order]

        data = {
            "image_rgb": image_rgb,
            "k_rgb": np.array(camera_info.k),
            "base_transform": self.transform_msg_to_matrix(base_transform.transform),
            "camera_transform": self.transform_msg_to_matrix(
                camera_transform.transform
            ),
            "joint_states": np.array(self._joint_state.position)[joint_map],
        }

        print("Saving", flush=True)
        with open(self._filename, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        exit()

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
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
