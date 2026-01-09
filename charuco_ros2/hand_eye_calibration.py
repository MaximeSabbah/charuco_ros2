"""
Offline ChArUco‑based camera and hand–eye calibration for ROS 2
==============================================================

This module provides a command‑line tool to estimate the rigid
transformation between a robot gripper (end effector) and an on‑board
camera using a ChArUco calibration target.  It operates purely on data
logged by the ``image_saver_calibration`` node and does not depend on
ROS 2 at runtime.  The procedure follows these steps:

* Define the ChArUco board geometry (number of squares, square size and
  marker size) to match your physical target.
* Detect ArUco markers and interpolate ChArUco corners in each image
  using ``cv2.aruco.CharucoDetector``.
* Recover the pose of the target relative to the camera by solving a
  PnP problem via ``cv2.solvePnP``.
* Invert recorded base→gripper transformations to obtain the required
  gripper→base poses.
* Run ``cv2.calibrateHandEye`` to solve for the constant camera→gripper
  transform.

After estimating the extrinsic, the tool performs a **verification
check**.  For each capture it reconstructs the pose of the ChArUco
target relative to the robot base by combining the estimated
camera→gripper transform, the recorded robot kinematics and the
observed board pose.  If the calibration target was stationary
throughout the acquisition, all reconstructed board poses should
coincide up to noise.  The tool therefore reports the root mean square
error (RMSE) of the translation (in metres) and orientation (in
degrees) of the predicted board poses relative to their mean.  Large
RMSE values indicate an inconsistent calibration, for example due to
insufficiently varied captures or incorrect board geometry.

Example usage:

.. code-block:: bash

    # Assuming you have run the image_saver_calibration node and have
    # collected image_data_*.pkl files in a directory called ``data``.
    ros2 run charuco_ros2 hand_eye_calibration.py data

The script will print the estimated rotation matrix and translation
vector mapping camera coordinates to gripper coordinates, followed by
a verification summary.
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def load_pickle_files(data_dir: Path) -> List[Path]:
    """Return a sorted list of pickle files in a directory."""
    return sorted(data_dir.glob("image_data_*.pkl"))


def detect_charuco_pose(
    image: np.ndarray,
    k_matrix: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    dist_coeffs: np.ndarray | None = None,
    min_corners: int = 4,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Detect ChArUco corners and estimate the target→camera pose.

    Parameters
    ----------
    image: numpy.ndarray
        BGR image containing the ChArUco board.
    k_matrix: numpy.ndarray
        3×3 camera intrinsic matrix.
    board: cv2.aruco.CharucoBoard
        Definition of the physical ChArUco board.
    dist_coeffs: numpy.ndarray | None, optional
        Distortion coefficients.  If omitted or ``None``, zero distortion
        is assumed.
    min_corners: int
        Minimum number of ChArUco corners required to attempt pose
        estimation.  Frames with fewer corners will be skipped.

    Returns
    -------
    (R, t): tuple of ndarray
        3×3 rotation matrix and 3×1 translation vector (in metres) that
        transform points from the target frame to the camera frame.
    None:
        If detection fails or not enough corners are visible.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    obj_pts = obj_pts.reshape(-1, 1, 3).astype(np.float32)
    img_pts = img_pts.reshape(-1, 1, 2).astype(np.float32)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        k_matrix.astype(np.float64),
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t


def invert_transform(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert a 4×4 homogeneous transformation matrix.

    Returns the rotation matrix and translation vector of the inverse.
    """
    R = H[:3, :3]
    t = H[:3, 3:4]
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def run_hand_eye_calibration(
    data_dir: Path,
    k_matrix_override: np.ndarray | None = None,
    dist_coeffs_override: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Perform hand–eye calibration from a directory of pickle files."""
    # Estimate the constant camera→gripper transform using ChArUco board detections and robot kinematics.
    files = load_pickle_files(data_dir)
    if not files:
        raise RuntimeError(f"No pickle files found in {data_dir}")
    nb_squares_x, nb_squares_y = 7, 5
    square_length = 0.054  # 54 mm
    marker_length = 0.040  # 40 mm
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard((nb_squares_x, nb_squares_y), square_length, marker_length, dictionary)
    R_gripper2base_list: List[np.ndarray] = []
    t_gripper2base_list: List[np.ndarray] = []
    R_target2cam_list: List[np.ndarray] = []
    t_target2cam_list: List[np.ndarray] = []
    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        base_transform = np.array(data["base_transform"], dtype=float)
        R_g2b, t_g2b = invert_transform(base_transform)
        R_gripper2base_list.append(R_g2b)
        t_gripper2base_list.append(t_g2b)
        image = data["image_rgb"]
        if k_matrix_override is not None:
            k_matrix = k_matrix_override
        else:
            k_matrix = np.array(data["k_rgb"], dtype=float).reshape(3, 3)
        pose = detect_charuco_pose(
            image,
            k_matrix,
            board,
            dist_coeffs=dist_coeffs_override if dist_coeffs_override is not None else None,
        )
        if pose is None:
            continue
        R_t2c, t_t2c = pose
        R_target2cam_list.append(R_t2c)
        t_target2cam_list.append(t_t2c)
    n = len(R_target2cam_list)
    if n < 3:
        raise RuntimeError(
            "At least three valid captures are required for hand–eye calibration; "
            f"only {n} were detected."
        )
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list,
        t_gripper2base_list,
        R_target2cam_list,
        t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    return R_cam2gripper, t_cam2gripper


def calibrate_camera_intrinsics(
    data_dir: Path,
    nb_squares_x: int = 7,
    nb_squares_y: int = 5,
    square_length: float = 0.054,
    marker_length: float = 0.040,
    min_corners: int = 6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate the camera intrinsic matrix and distortion coefficients.

    This function collects ChArUco observations from all pickle files in
    ``data_dir`` and uses ``cv2.aruco.calibrateCameraCharuco`` to estimate the
    intrinsic camera matrix and distortion coefficients.  At least three
    valid views with a sufficient number of detected corners are required.

    Parameters
    ----------
    data_dir: pathlib.Path
        Directory containing ``image_data_*.pkl`` files.
    nb_squares_x, nb_squares_y: int, optional
        Number of squares along the X and Y dimensions of the ChArUco board.
    square_length, marker_length: float, optional
        Dimensions of the squares and markers in metres.
    min_corners: int, optional
        Minimum number of ChArUco corners required to include a frame in the
        calibration.

    Returns
    -------
    (K, D, reproj_err): tuple
        ``K`` is a 3×3 camera matrix, ``D`` is the distortion coefficient
        vector, and ``reproj_err`` is the average reprojection error reported
        by OpenCV.

    Raises
    ------
    RuntimeError
        If fewer than three valid frames are available.
    """
    files = load_pickle_files(data_dir)
    if not files:
        raise RuntimeError(f"No pickle files found in {data_dir}")
    # Create Charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard(
        (nb_squares_x, nb_squares_y), square_length, marker_length, dictionary
    )
    # Precompute all chessboard corner positions (3D) for the board.
    chessboard_3d = board.getChessboardCorners().astype(np.float32)
    object_points_list: List[np.ndarray] = []
    image_points_list: List[np.ndarray] = []
    image_size: Tuple[int, int] | None = None
    valid_count = 0
    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        image = data["image_rgb"]
        if image_size is None:
            h, w = image.shape[:2]
            image_size = (w, h)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, *_ = detector.detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) < min_corners:
            continue
        # Map detected charuco IDs to their 3D board coordinates
        obj_pts = []
        img_pts = []
        for corner, cid in zip(charuco_corners, charuco_ids):
            idx = int(cid)
            if idx < len(chessboard_3d):
                obj_pts.append(chessboard_3d[idx])
                img_pts.append(corner[0])
        if len(obj_pts) < min_corners:
            continue
        object_points_list.append(np.array(obj_pts, dtype=np.float32))
        image_points_list.append(np.array(img_pts, dtype=np.float32))
        valid_count += 1
    if valid_count < 3:
        raise RuntimeError(
            "At least three valid captures with sufficient ChArUco corners "
            f"are required for intrinsic calibration; only {valid_count} were detected."
        )
    # Perform camera calibration using the collected correspondences.  We do not fix
    # intrinsic parameters or aspect ratio; flags can be extended if desired.
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list,
        image_points_list,
        image_size,
        None,
        None,
    )
    return camera_matrix.astype(np.float64), dist_coeffs.reshape(-1, 1).astype(np.float64), float(ret)


def verify_hand_eye_calibration(
    data_dir: Path,
    R_cam2gripper: np.ndarray,
    t_cam2gripper: np.ndarray,
    nb_squares_x: int = 7,
    nb_squares_y: int = 5,
    square_length: float = 0.054,
    marker_length: float = 0.040,
) -> Tuple[float, float]:
    """Verify the consistency of a hand–eye calibration.

    After calibration, this function reconstructs the pose of the target
    relative to the robot base for each capture using the estimated
    extrinsic.  It then measures the dispersion of these reconstructed
    poses.  If the camera→gripper transform is accurate and the target
    was stationary, all reconstructed poses should coincide.  The
    returned RMSE values quantify the translation and orientation
    differences; large values indicate a problem with the calibration.

    Returns
    -------
    (rmse_t, rmse_r): tuple of float
        Root mean square error of the translation (in metres) and the
        orientation (in degrees) of the predicted board poses relative
        to their mean.
    """
    files = load_pickle_files(data_dir)
    if not files:
        raise RuntimeError(f"No pickle files found in {data_dir}")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard((nb_squares_x, nb_squares_y), square_length, marker_length, dictionary)
    R_g2cam = R_cam2gripper.T
    t_g2cam = -R_cam2gripper.T @ t_cam2gripper
    translations: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        base_transform = np.array(data["base_transform"], dtype=float)
        R_b_g = base_transform[:3, :3]
        t_b_g = base_transform[:3, 3:4]
        R_b_c = R_b_g @ R_g2cam
        t_b_c = R_b_g @ t_g2cam + t_b_g
        image = data["image_rgb"]
        k_matrix = np.array(data["k_rgb"], dtype=float).reshape(3, 3)
        pose = detect_charuco_pose(image, k_matrix, board)
        if pose is None:
            continue
        R_t2c, t_t2c = pose
        R_c2t = R_t2c.T
        t_c2t = -R_t2c.T @ t_t2c
        R_b_t = R_b_c @ R_c2t
        t_b_t = R_b_c @ t_c2t + t_b_c
        translations.append(t_b_t.reshape(3))
        rotations.append(R_b_t)
    if len(translations) < 2:
        raise RuntimeError("Verification requires at least two valid captures.")
    translations_np = np.stack(translations)
    t_mean = translations_np.mean(axis=0)
    R_ref = rotations[0]
    angle_errors = []
    for R_i in rotations:
        delta_R = R_ref.T @ R_i
        # Clamp trace to avoid numerical issues
        angle = np.arccos(max(min((np.trace(delta_R) - 1.0) / 2.0, 1.0), -1.0))
        angle_deg = np.degrees(angle)
        angle_errors.append(angle_deg)
    t_errors = np.linalg.norm(translations_np - t_mean, axis=1)
    rmse_t = float(np.sqrt(np.mean(t_errors ** 2)))
    rmse_r = float(np.sqrt(np.mean(np.array(angle_errors) ** 2)))
    return rmse_t, rmse_r


def average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
    """Compute the average of a list of rotation matrices.

    This function computes the mean rotation matrix by
    summing all rotation matrices and projecting the result
    back onto the space of orthonormal matrices via a singular
    value decomposition (SVD).  If the resulting rotation
    matrix has a negative determinant, the last column of the
    left singular vectors is flipped to ensure a proper
    rotation (determinant +1).

    Parameters
    ----------
    rotations: list of np.ndarray
        List of 3×3 rotation matrices.

    Returns
    -------
    np.ndarray
        The 3×3 average rotation matrix.
    """
    if not rotations:
        raise ValueError("Rotation list must not be empty")
    S = np.zeros((3, 3), dtype=float)
    for R in rotations:
        S += R
    # Compute the orthogonal factor of S via SVD
    U, _, Vt = np.linalg.svd(S)
    R_mean = U @ Vt
    # Ensure a proper rotation (determinant +1)
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    return R_mean


def project_board_axes(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    k_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length: float = 0.05,
    color: tuple[int, int, int] | None = None,
) -> None:
    """Draw the three coordinate axes of a ChArUco board pose on an image.

    The axes are drawn at the origin of the board coordinate system and
    extend for ``axis_length`` metres along the X (red), Y (green) and
    Z (blue) axes.  If a specific BGR ``color`` is provided, a single
    colour will be used for all axes; otherwise, OpenCV's default
    colours are used.  The function modifies the input image in place.

    Parameters
    ----------
    image: numpy.ndarray
        The colour image onto which the axes will be drawn.
    board: cv2.aruco.CharucoBoard
        The ChArUco board definition (unused here but kept for
        consistency).
    k_matrix: numpy.ndarray
        Camera intrinsic matrix.
    dist_coeffs: numpy.ndarray
        Distortion coefficients (K1…K5).  A zero vector may be
        provided if no distortion is assumed.
    rvec: numpy.ndarray
        Rotation vector (Rodrigues) of the board→camera pose.
    tvec: numpy.ndarray
        Translation vector of the board→camera pose.
    axis_length: float, optional
        Length of each axis in metres.  Defaults to 0.05 m (5 cm).
    color: tuple[int, int, int] | None, optional
        Optional BGR colour for the axes.  If None, the axes are
        coloured red, green and blue for X, Y and Z respectively.
    """
    # Define the three axis endpoints in the board coordinate system
    axis_pts = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],  # X axis
        [0, axis_length, 0],  # Y axis
        [0, 0, axis_length],  # Z axis
    ]).reshape(-1, 3)
    # Project the axis points to image coordinates
    img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, k_matrix, dist_coeffs)
    img_pts = img_pts.reshape(-1, 2).astype(int)
    origin = tuple(img_pts[0])
    # Draw lines for each axis
    colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: red, green, blue
    if color is not None:
        colours = [color] * 3
    for i in range(3):
        pt = tuple(img_pts[i + 1])
        cv2.line(image, origin, pt, colours[i], 2, cv2.LINE_AA)


def draw_predicted_board_outline(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    k_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    *,
    square_length: float,
    colour: tuple[int, int, int] = (0, 255, 255),
) -> None:
    """Draw the outer rectangle of the predicted board pose on an image.

    This helper projects the four outer corners of the ChArUco board
    using the supplied pose and draws a polyline connecting them.

    Parameters
    ----------
    image: numpy.ndarray
        The image on which to draw.
    board: cv2.aruco.CharucoBoard
        The ChArUco board definition.
    k_matrix: numpy.ndarray
        Camera intrinsic matrix.
    dist_coeffs: numpy.ndarray
        Distortion coefficients.
    rvec: numpy.ndarray
        Rotation vector (Rodrigues) of the board→camera pose.
    tvec: numpy.ndarray
        Translation vector of the board→camera pose.
    colour: tuple[int, int, int], optional
        BGR colour of the outline.  Defaults to yellow.
    """
    # Compute the 3D coordinates of the four board corners (top-left, top-right, bottom-right, bottom-left)
    w, h = board.getChessboardSize()
    # Board corners are defined in the board coordinate system.  X axis points right, Y axis points up.
    # The corners in OpenCV's CharucoBoard correspond to indices (0,0) at the top-left and (w-1,h-1) at the bottom-right.
    top_left = np.array([0, 0, 0], dtype=np.float32)
    top_right = np.array([w * square_length, 0, 0], dtype=np.float32)
    bottom_right = np.array([w * square_length, h * square_length, 0], dtype=np.float32)
    bottom_left = np.array([0, h * square_length, 0], dtype=np.float32)
    corners_3d = np.stack([top_left, top_right, bottom_right, bottom_left])
    img_pts, _ = cv2.projectPoints(corners_3d, rvec, tvec, k_matrix, dist_coeffs)
    pts = img_pts.reshape(-1, 2).astype(int)
    cv2.polylines(image, [pts], isClosed=True, color=colour, thickness=2, lineType=cv2.LINE_AA)


def create_visualizations(
    data_dir: Path,
    R_cam2gripper: np.ndarray,
    t_cam2gripper: np.ndarray,
    nb_squares_x: int = 7,
    nb_squares_y: int = 5,
    square_length: float = 0.054,
    marker_length: float = 0.040,
    output_dir: Path | None = None,
    ) -> None:
    """Generate visual verification plots and overlay images.

    This function produces two types of outputs:

    1. For each captured frame, it overlays the detected ChArUco corners
       and draws both the measured board coordinate frame and the
       predicted board coordinate frame on top of the original image.
       The predicted board pose is computed from the estimated
       camera→gripper transform, the recorded robot kinematics and the
       mean board pose across captures.  These images are saved to
       ``output_dir`` as PNG files.

    2. A 3D plot of the reconstructed board poses (target→base) for all
       valid captures.  Each pose is visualised as a coordinate frame
       anchored at the board origin.  The average pose is drawn in
       heavier lines.  The plot is saved as ``board_poses.png`` in
       ``output_dir``.

    Parameters
    ----------
    data_dir: pathlib.Path
        Directory containing ``image_data_*.pkl`` files.
    R_cam2gripper, t_cam2gripper: numpy.ndarray
        Estimated camera→gripper rotation matrix and translation vector.
    nb_squares_x, nb_squares_y: int, optional
        Number of squares along the X and Y dimensions of the ChArUco board.
    square_length, marker_length: float, optional
        Dimensions of the squares and markers in metres.
    output_dir: pathlib.Path | None, optional
        Directory where visualisations will be saved.  If ``None``, a
        subdirectory called ``visuals`` inside ``data_dir`` will be
        created.
    """
    import matplotlib
    matplotlib.use("Agg")  # Use non‑interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    files = load_pickle_files(data_dir)
    if not files:
        raise RuntimeError(f"No pickle files found in {data_dir}")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard((nb_squares_x, nb_squares_y), square_length, marker_length, dictionary)
    # Compute gripper→camera from the extrinsic
    R_g2c = R_cam2gripper.T
    t_g2c = -R_cam2gripper.T @ t_cam2gripper
    rotations_b_t: List[np.ndarray] = []
    translations_b_t: List[np.ndarray] = []
    detections: List[Tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    # Gather board→base poses and detection results
    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        image = data["image_rgb"].copy()
        k_matrix = np.array(data["k_rgb"], dtype=float).reshape(3, 3)
        base_transform = np.array(data["base_transform"], dtype=float)
        # Compute base→camera transform (camera→base is R_b_c, t_b_c; base→camera is its inverse)
        R_b_g = base_transform[:3, :3]
        t_b_g = base_transform[:3, 3:4]
        R_b_c = R_b_g @ R_g2c
        t_b_c = R_b_g @ t_g2c + t_b_g
        # board pose from detection
        pose = detect_charuco_pose(image, k_matrix, board)
        if pose is None:
            continue
        R_t2c, t_t2c = pose  # board→camera
        # Compute board→base via composition: board→base = board→camera followed by camera→base
        R_c2t = R_t2c.T
        t_c2t = -R_t2c.T @ t_t2c
        R_b_t = R_b_c @ R_c2t
        t_b_t = R_b_c @ t_c2t + t_b_c
        rotations_b_t.append(R_b_t)
        translations_b_t.append(t_b_t.reshape(3))
        # Store detection details for later visualisation
        detections.append((path, image, k_matrix, R_t2c, t_t2c, R_b_c, t_b_c))
    if len(rotations_b_t) < 2:
        raise RuntimeError("Visualisation requires at least two valid captures.")
    # Compute mean board→base pose
    R_b_t_mean = average_rotations(rotations_b_t)
    t_b_t_mean = np.mean(np.stack(translations_b_t), axis=0).reshape(3, 1)
    # Prepare output directory
    if output_dir is None:
        output_dir = data_dir / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Loop through detections and produce overlay images
    for i, (path, image, k_matrix, R_t2c, t_t2c, R_b_c, t_b_c) in enumerate(detections):
        # Draw detected ChArUco corners and axes (ground truth) on a copy of the image
        img_det = image.copy()
        # Re-run Charuco detection to obtain corners and IDs for drawing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        charuco_detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(img_det, charuco_corners, charuco_ids, (0, 255, 0))
            # Draw detected markers if available
            if marker_corners is not None:
                cv2.aruco.drawDetectedMarkers(img_det, marker_corners, marker_ids)
        # Draw the measured coordinate frame on the detected pose
        rvec_det, _ = cv2.Rodrigues(R_t2c)
        tvec_det = t_t2c.copy()
        project_board_axes(img_det, board, k_matrix, np.zeros((5, 1), dtype=float), rvec_det, tvec_det, axis_length=0.04)
        # Draw predicted coordinate frame using mean board pose and base→camera
        # Compute base→camera inverse (camera→base is R_b_c; base→camera is its inverse)
        R_c_b = R_b_c.T
        t_c_b = -R_b_c.T @ t_b_c
        # Predicted board→camera: board→base then base→camera
        R_pred = R_c_b @ R_b_t_mean
        t_pred = R_c_b @ t_b_t_mean + t_c_b
        rvec_pred, _ = cv2.Rodrigues(R_pred)
        tvec_pred = t_pred.copy()
        project_board_axes(img_det, board, k_matrix, np.zeros((5, 1), dtype=float), rvec_pred, tvec_pred, axis_length=0.04, color=(255, 255, 0))
        # Optionally draw the predicted board outline
        draw_predicted_board_outline(
            img_det,
            board,
            k_matrix,
            np.zeros((5, 1), dtype=float),
            rvec_pred,
            tvec_pred,
            square_length=square_length,
            colour=(255, 255, 0),
        )
        # Save the annotated image
        out_path = output_dir / f"frame_{i:03d}.png"
        cv2.imwrite(str(out_path), img_det)
    # Create a 3D plot of all board poses in base coordinates
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Plot each pose as a coordinate frame
    def draw_frame(ax, origin: np.ndarray, R: np.ndarray, length: float = 0.05, lw: float = 1.0):
        # Axes: X (red), Y (green), Z (blue)
        colours = ['r', 'g', 'b']
        for i in range(3):
            axis = R[:, i] * length
            ax.plot(
                [origin[0], origin[0] + axis[0]],
                [origin[1], origin[1] + axis[1]],
                [origin[2], origin[2] + axis[2]],
                color=colours[i],
                linewidth=lw,
            )
    # Draw each reconstructed pose
    for R_b_t, t_b_t in zip(rotations_b_t, translations_b_t):
        draw_frame(ax, t_b_t, R_b_t, length=0.04, lw=1.0)
    # Draw the mean pose with thicker lines
    draw_frame(ax, t_b_t_mean.flatten(), R_b_t_mean, length=0.05, lw=2.0)
    # Scatter the origins of all poses
    poses_np = np.stack(translations_b_t)
    ax.scatter(poses_np[:, 0], poses_np[:, 1], poses_np[:, 2], c='k', marker='o', s=8, label='Board poses')
    # Label axes and set equal aspect ratios
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Reconstructed board poses (base frame)')
    # Set equal scaling
    max_range = np.array([
        poses_np[:, 0].max() - poses_np[:, 0].min(),
        poses_np[:, 1].max() - poses_np[:, 1].min(),
        poses_np[:, 2].max() - poses_np[:, 2].min(),
    ]).max() / 2.0
    mid_x = (poses_np[:, 0].max() + poses_np[:, 0].min()) / 2.0
    mid_y = (poses_np[:, 1].max() + poses_np[:, 1].min()) / 2.0
    mid_z = (poses_np[:, 2].max() + poses_np[:, 2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig_path = output_dir / "board_poses.png"
    fig.savefig(fig_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate camera→gripper transform from ChArUco images and robot logs, "
            "optionally estimate the camera intrinsics, and verify the consistency "
            "of the calibration."
        )
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help=(
            "Directory containing image_data_*.pkl files produced by "
            "image_saver_calibration.py"
        ),
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Generate visual verification outputs.  Annotated images with detected "
            "and predicted board poses will be saved to a 'visuals' subdirectory, "
            "and a 3D plot of reconstructed board poses will be created.  This "
            "option requires Matplotlib to be available."
        ),
    )
    parser.add_argument(
        "--estimate-intrinsics",
        action="store_true",
        help=(
            "Estimate the camera intrinsic matrix and distortion coefficients "
            "from the ChArUco observations before running hand–eye calibration. "
            "If set, the estimated intrinsics will be printed and used for the "
            "hand–eye calibration instead of the intrinsics contained in the "
            "pickle files."
        ),
    )
    args = parser.parse_args()
    k_override = None
    dist_coeffs = None
    if args.estimate_intrinsics:
        try:
            k_override, dist_coeffs, reproj_err = calibrate_camera_intrinsics(args.data_dir)
            print("Estimated camera intrinsics:")
            print("Camera matrix (K):")
            print(k_override)
            print("Distortion coefficients (D):")
            print(dist_coeffs.flatten())
            print(f"Reprojection error: {reproj_err:.6f} pixels")
        except RuntimeError as e:
            print(f"Intrinsic calibration failed: {e}\nContinuing with provided intrinsics...")
            k_override = None
            dist_coeffs = None
    R_cam2gripper, t_cam2gripper = run_hand_eye_calibration(
        args.data_dir,
        k_matrix_override=k_override,
        dist_coeffs_override=(dist_coeffs if args.estimate_intrinsics else None),
    )
    print("Estimated camera→gripper transform:")
    print("Rotation (R_cam2gripper):")
    print(R_cam2gripper)
    print("Translation (t_cam2gripper):")
    print(t_cam2gripper.flatten())
    try:
        rmse_t, rmse_r = verify_hand_eye_calibration(
            args.data_dir, R_cam2gripper, t_cam2gripper
        )
        print()
        print("Verification summary:")
        print(
            f"Translation RMSE: {rmse_t:.4f} m; Orientation RMSE: {rmse_r:.2f} deg "
            "(lower is better)"
        )
    except RuntimeError as e:
        print(f"Verification skipped: {e}")

    # Generate visualisations if requested
    if args.visualize:
        try:
            create_visualizations(
                args.data_dir,
                R_cam2gripper,
                t_cam2gripper,
                nb_squares_x=7,
                nb_squares_y=5,
                square_length=0.054,
                marker_length=0.040,
                output_dir=None,
            )
            print()
            print(
                "Visualisations saved to the 'visuals' directory within the data directory. "
                "Use these images to inspect detected and predicted board poses and the "
                "3D distribution of board poses in the base frame."
            )
        except Exception as exc:
            print(f"Visualisation generation failed: {exc}")


if __name__ == "__main__":
    main()