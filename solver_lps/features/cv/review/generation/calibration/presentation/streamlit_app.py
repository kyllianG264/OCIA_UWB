"""
Manual basketball court calibration using the exact terrain image as target.

Two-camera workflow:
1. Click a point on left or right video frame.
2. Click the corresponding point on the terrain image.
3. The pair is stored for the selected camera only.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT_DIR_PATH not in sys.path:
    sys.path.insert(0, ROOT_DIR_PATH)

from solver_lps.features.cv.review.generation.calibration.data.calibration_output import (
    save_calibration,
)
from solver_lps.features.cv.review.generation.calibration.data.ground_input import (
    load_ground_input,
)
from solver_lps.features.cv.review.generation.calibration.data.video_input import (
    extract_frame,
)
from solver_lps.features.cv.review.generation.calibration.domain.distortion import (
    apply_radial_correction_to_points,
    apply_view_transform_to_points,
    compute_undistort_view_transform,
    invert_radial_correction_to_points,
    invert_view_transform_to_points,
    optimize_distortion,
    remap_frame_with_distortion,
)
from solver_lps.features.cv.review.generation.calibration.domain.homography import (
    bounds_from_points,
    compute_homography,
    compute_split_y,
)
from solver_lps.features.cv.review.data.session_assets import DEFAULT_SPORT

st.set_page_config(page_title="NeptuVision - Calibration", page_icon="target", layout="wide")

try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

LEFT_COLOR = (59, 130, 246)
RIGHT_COLOR = (239, 68, 68)
TEXT_BG = (0, 0, 0)
TEXT_FG = (255, 230, 0)
UNDISTORT_VIEW_MARGIN_PX = 12.0

ENV_SPORT = os.environ.get("SOLVER_SPORT") or DEFAULT_SPORT
ENV_LEFT_VIDEO = os.environ.get("SOLVER_LEFT_VIDEO")
ENV_RIGHT_VIDEO = os.environ.get("SOLVER_RIGHT_VIDEO")
ENV_OUTPUT_DIR = os.environ.get("SOLVER_CALIBRATION_OUTPUT_DIR")
try:
    GROUND_INPUT = load_ground_input(ENV_SPORT)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
DEFAULT_LEFT_VIDEO_PATH = GROUND_INPUT.default_left_video_path
DEFAULT_RIGHT_VIDEO_PATH = GROUND_INPUT.default_right_video_path
terrain_path = GROUND_INPUT.terrain_path
terrain_rgb = GROUND_INPUT.terrain_rgb
terrain_w = GROUND_INPUT.terrain_width
terrain_h = GROUND_INPUT.terrain_height



def click_widget(image_rgb: np.ndarray, display_width: int, key: str) -> Optional[tuple[int, int]]:
    original_h, original_w = image_rgb.shape[:2]
    scale = display_width / original_w
    display_h = int(original_h * scale)
    resized = cv2.resize(image_rgb, (display_width, display_h))
    display_image = Image.fromarray(resized)

    if HAS_CLICK:
        coord = streamlit_image_coordinates(display_image, key=key)
        if coord is None:
            return None
        return int(coord["x"] / scale), int(coord["y"] / scale)

    st.image(display_image, use_container_width=True)
    col_x, col_y = st.columns(2)
    raw_x = col_x.number_input("X", 0, original_w, original_w // 2, key=f"{key}_x")
    raw_y = col_y.number_input("Y", 0, original_h, original_h // 2, key=f"{key}_y")
    if st.button("Validate point", key=f"{key}_ok"):
        return int(raw_x), int(raw_y)
    return None


def _draw_label(image: np.ndarray, label: str, x_value: int, y_value: int, color: tuple[int, int, int]) -> None:
    cv2.circle(image, (x_value, y_value), 18, color, -1)
    cv2.circle(image, (x_value, y_value), 18, (255, 255, 255), 2)
    cv2.putText(image, label, (x_value + 10, y_value - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_BG, 4, cv2.LINE_AA)
    cv2.putText(image, label, (x_value + 10, y_value - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def annotate_video(image_rgb: np.ndarray, video_points: list[list[int]], prefix: str, color: tuple[int, int, int], pending_point: Optional[list[int]] = None) -> np.ndarray:
    output = image_rgb.copy()
    for index, point in enumerate(video_points):
        _draw_label(output, f"{prefix}{index + 1}", int(point[0]), int(point[1]), color)
    if pending_point is not None:
        _draw_label(output, f"{prefix}?", int(pending_point[0]), int(pending_point[1]), color)
    return output


def annotate_distortion_lines(image_rgb: np.ndarray, lines, current_line, prefix: str, color: tuple[int, int, int], distortion: Optional[dict] = None) -> np.ndarray:
    base = remap_frame_with_distortion(image_rgb, distortion)
    output = base.copy()
    for line_index, line in enumerate(lines):
        corrected_line = apply_radial_correction_to_points(line, distortion) if distortion else line
        for point_index, point in enumerate(corrected_line):
            label = f"{prefix}D{line_index + 1}.{point_index + 1}"
            _draw_label(output, label, int(round(point[0])), int(round(point[1])), color)
        if len(corrected_line) >= 2:
            poly = np.array([[int(round(point[0])), int(round(point[1]))] for point in corrected_line], dtype=np.int32)
            cv2.polylines(output, [poly], False, color, 2, cv2.LINE_AA)
    if current_line:
        corrected_current = apply_radial_correction_to_points(current_line, distortion) if distortion else current_line
        for point_index, point in enumerate(corrected_current):
            label = f"{prefix}C{point_index + 1}"
            _draw_label(output, label, int(round(point[0])), int(round(point[1])), color)
        if len(corrected_current) >= 2:
            poly = np.array([[int(round(point[0])), int(round(point[1]))] for point in corrected_current], dtype=np.int32)
            cv2.polylines(output, [poly], False, color, 2, cv2.LINE_AA)
    return output


def annotate_terrain(
    left_terrain_points: list[list[float]],
    right_terrain_points: list[list[float]],
    pending_camera: Optional[str] = None,
    pending_terrain_point: Optional[list[int]] = None,
) -> np.ndarray:
    output = terrain_rgb.copy()
    for index, point in enumerate(left_terrain_points):
        _draw_label(output, f"L{index + 1}", int(round(point[0])), int(round(point[1])), LEFT_COLOR)
    for index, point in enumerate(right_terrain_points):
        _draw_label(output, f"R{index + 1}", int(round(point[0])), int(round(point[1])), RIGHT_COLOR)
    if pending_camera and pending_terrain_point is not None:
        prefix = "L" if pending_camera == "g" else "R"
        color = LEFT_COLOR if pending_camera == "g" else RIGHT_COLOR
        _draw_label(output, f"{prefix}?", int(pending_terrain_point[0]), int(pending_terrain_point[1]), color)
    return output


def preview_projection(homography: np.ndarray, video_points: list[list[int]], terrain_points: list[list[float]], inlier_mask: list[int], color: tuple[int, int, int], prefix: str) -> np.ndarray:
    preview = terrain_rgb.copy()
    projected = cv2.perspectiveTransform(np.array(video_points, dtype=np.float32).reshape(-1, 1, 2), homography).reshape(-1, 2)
    for index, target in enumerate(terrain_points):
        tx, ty = int(round(target[0])), int(round(target[1]))
        px, py = int(round(projected[index][0])), int(round(projected[index][1]))
        link_color = (60, 220, 120) if inlier_mask[index] else (255, 80, 80)
        _draw_label(preview, f"{prefix}{index + 1}", tx, ty, color)
        cv2.line(preview, (tx, ty), (px, py), link_color, 3)
        cv2.rectangle(preview, (px - 8, py - 8), (px + 8, py + 8), link_color, 3)
    return preview


def draw_video_sampling_grid(frame: np.ndarray, homography: np.ndarray) -> np.ndarray:
    preview = terrain_rgb.copy()
    for row_index in range(7):
        for col_index in range(11):
            sample_x = int((frame.shape[1] - 1) * col_index / 10.0)
            sample_y = int((frame.shape[0] - 1) * row_index / 6.0)
            point = np.array([[[sample_x, sample_y]]], dtype=np.float32)
            projected = cv2.perspectiveTransform(point, homography)[0][0]
            px = int(round(projected[0]))
            py = int(round(projected[1]))
            if 0 <= px < terrain_w and 0 <= py < terrain_h:
                cv2.circle(preview, (px, py), 5, (100, 200, 255), -1)
    return preview


def init_state() -> None:
    defaults = {
        "frame_g": None,
        "frame_d": None,
        "vid_pts_g": [],
        "vid_pts_d": [],
        "ter_pts_g": [],
        "ter_pts_d": [],
        "pending_camera": None,
        "pending_video_point": None,
        "prev_click_video_g": None,
        "prev_click_video_d": None,
        "prev_click_terrain": None,
        "distortion_lines_g": [],
        "distortion_lines_d": [],
        "distortion_current_line_g": [],
        "distortion_current_line_d": [],
        "distortion_g": None,
        "distortion_d": None,
        "undistort_view_g": None,
        "undistort_view_d": None,
        "distortion_mode": False,
        "prev_click_distortion_g": None,
        "prev_click_distortion_d": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

st.title("Manual basketball court calibration")
st.markdown(
    """
Workflow:
1. click a point on the left or right video frame
2. click the matching point on the terrain image
3. the pair is stored for that camera only
"""
)

if not HAS_CLICK:
    st.warning("Install `streamlit-image-coordinates` for click selection. Manual coordinate mode is active.")

with st.sidebar:
    st.header("Settings")
    left_video_path = st.text_input("Left video", value=str(ENV_LEFT_VIDEO or DEFAULT_LEFT_VIDEO_PATH), key="sid_vg")
    right_video_path = st.text_input("Right video", value=str(ENV_RIGHT_VIDEO or DEFAULT_RIGHT_VIDEO_PATH), key="sid_vd")
    extract_second = st.number_input("Extraction timestamp (s)", 0.0, value=5.0, step=1.0)
    output_dir = st.text_input("Output directory", str(ENV_OUTPUT_DIR or GROUND_INPUT.default_output_dir))

    if st.button("Extract frames", type="primary"):
        success = True
        for label, path, frame_key, side_key in [
            ("left", left_video_path, "frame_g", "g"),
            ("right", right_video_path, "frame_d", "d"),
        ]:
            if not path or not os.path.isfile(path):
                st.error(f"Missing {label} video: {path}")
                success = False
                continue
            try:
                st.session_state[frame_key] = extract_frame(path, extract_second)
                st.session_state[f"vid_pts_{side_key}"] = []
                st.session_state[f"ter_pts_{side_key}"] = []
                st.session_state[f"undistort_view_{side_key}"] = None
            except Exception as exc:
                st.error(str(exc))
                success = False
        st.session_state["pending_camera"] = None
        st.session_state["pending_video_point"] = None
        st.session_state["prev_click_video_g"] = None
        st.session_state["prev_click_video_d"] = None
        st.session_state["prev_click_terrain"] = None
        if success:
            st.success("Frames extracted")


left_frame = st.session_state["frame_g"]
right_frame = st.session_state["frame_d"]
left_video_points = st.session_state["vid_pts_g"]
right_video_points = st.session_state["vid_pts_d"]
left_terrain_points = st.session_state["ter_pts_g"]
right_terrain_points = st.session_state["ter_pts_d"]
pending_camera = st.session_state["pending_camera"]
pending_video_point = st.session_state["pending_video_point"]
distortion_left = st.session_state["distortion_g"]
distortion_right = st.session_state["distortion_d"]
undistort_view_left = None
undistort_view_right = None
if left_frame is not None:
    undistort_view_left = st.session_state["undistort_view_g"] or compute_undistort_view_transform(left_frame.shape[1], left_frame.shape[0], distortion_left)
if left_frame is not None:
    st.session_state["undistort_view_g"] = undistort_view_left
if right_frame is not None:
    undistort_view_right = st.session_state["undistort_view_d"] or compute_undistort_view_transform(right_frame.shape[1], right_frame.shape[0], distortion_right)
if right_frame is not None:
    st.session_state["undistort_view_d"] = undistort_view_right

left_video_points_corrected = left_video_points
right_video_points_corrected = right_video_points

left_result = None
right_result = None
if len(left_video_points) >= 4 and len(left_terrain_points) >= 4:
    try:
        left_result = compute_homography(left_video_points_corrected, left_terrain_points)
    except Exception:
        left_result = None
if len(right_video_points) >= 4 and len(right_terrain_points) >= 4:
    try:
        right_result = compute_homography(right_video_points_corrected, right_terrain_points)
    except Exception:
        right_result = None

terrain_section, status_section = st.columns([1.15, 0.85], gap="large")

with terrain_section:
    st.subheader("Terrain reference")
    terrain_image = annotate_terrain(left_terrain_points, right_terrain_points)
    if pending_camera and pending_video_point is not None:
        help_text = f"Pending pair for {'left' if pending_camera == 'g' else 'right'} camera: click the matching terrain point."
        st.caption(help_text)
    terrain_click = click_widget(terrain_image, 520, key="terrain_main")
    if terrain_click is not None and list(terrain_click) != st.session_state["prev_click_terrain"]:
        st.session_state["prev_click_terrain"] = list(terrain_click)
        if pending_camera == "g" and pending_video_point is not None:
            st.session_state["vid_pts_g"] = left_video_points + [list(pending_video_point)]
            st.session_state["ter_pts_g"] = left_terrain_points + [list(terrain_click)]
            st.session_state["pending_camera"] = None
            st.session_state["pending_video_point"] = None
            st.rerun()
        if pending_camera == "d" and pending_video_point is not None:
            st.session_state["vid_pts_d"] = right_video_points + [list(pending_video_point)]
            st.session_state["ter_pts_d"] = right_terrain_points + [list(terrain_click)]
            st.session_state["pending_camera"] = None
            st.session_state["pending_video_point"] = None
            st.rerun()
    if pending_camera is None:
        st.caption("Click a point on either video frame, then click its matching point here.")

with status_section:
    st.subheader("Calibration status")
    st.toggle("Distortion calibration mode", key="distortion_mode")
    if left_result is not None:
        st.metric("Left RMS", f"{left_result[2]:.2f} px")
        st.write(f"Left inliers: {sum(left_result[1])}/{len(left_result[1])}")
    else:
        st.write(f"Left pairs: {len(left_video_points)}")
    if right_result is not None:
        st.metric("Right RMS", f"{right_result[2]:.2f} px")
        st.write(f"Right inliers: {sum(right_result[1])}/{len(right_result[1])}")
    else:
        st.write(f"Right pairs: {len(right_video_points)}")
    if pending_camera is not None:
        st.info(f"Pending terrain click for {'left' if pending_camera == 'g' else 'right'} camera.")
    if distortion_left is not None:
        st.write(f"Left distortion: k1={distortion_left['k1']:.5f}, k2={distortion_left['k2']:.5f}, enabled={distortion_left['enabled']}")
    if distortion_right is not None:
        st.write(f"Right distortion: k1={distortion_right['k1']:.5f}, k2={distortion_right['k2']:.5f}, enabled={distortion_right['enabled']}")


st.divider()
st.subheader("Video frames")

if left_frame is None or right_frame is None:
    st.info("Extract both frames from the sidebar to start calibration.")
else:
    if st.session_state["distortion_mode"]:
        left_image = annotate_distortion_lines(
            left_frame,
            st.session_state["distortion_lines_g"],
            st.session_state["distortion_current_line_g"],
            "L",
            LEFT_COLOR,
            None,
        )
        right_image = annotate_distortion_lines(
            right_frame,
            st.session_state["distortion_lines_d"],
            st.session_state["distortion_current_line_d"],
            "R",
            RIGHT_COLOR,
            None,
        )
    else:
        left_image = annotate_video(
            remap_frame_with_distortion(left_frame, distortion_left, undistort_view_left),
            left_video_points,
            "L",
            LEFT_COLOR,
            pending_video_point if pending_camera == "g" and pending_video_point is not None else None,
        )
        right_image = annotate_video(
            remap_frame_with_distortion(right_frame, distortion_right, undistort_view_right),
            right_video_points,
            "R",
            RIGHT_COLOR,
            pending_video_point if pending_camera == "d" and pending_video_point is not None else None,
        )

    st.markdown("**Left camera**")
    left_click = click_widget(left_image, 980, key="video_left")
    if left_click is not None and list(left_click) != st.session_state["prev_click_video_g"]:
        st.session_state["prev_click_video_g"] = list(left_click)
        if st.session_state["distortion_mode"]:
            if list(left_click) != st.session_state["prev_click_distortion_g"]:
                st.session_state["prev_click_distortion_g"] = list(left_click)
                st.session_state["distortion_current_line_g"] = st.session_state["distortion_current_line_g"] + [list(left_click)]
        else:
            st.session_state["pending_camera"] = "g"
            st.session_state["pending_video_point"] = list(left_click)
        st.rerun()

    left_controls = st.columns(4)
    if left_controls[0].button("Undo last left pair", disabled=len(left_video_points) == 0):
        st.session_state["vid_pts_g"] = left_video_points[:-1]
        st.session_state["ter_pts_g"] = left_terrain_points[:-1]
        st.rerun()
    if left_controls[1].button("Reset left camera"):
        st.session_state["vid_pts_g"] = []
        st.session_state["ter_pts_g"] = []
        if st.session_state["pending_camera"] == "g":
            st.session_state["pending_camera"] = None
            st.session_state["pending_video_point"] = None
        st.rerun()
    if left_controls[2].button("Validate left line", disabled=len(st.session_state["distortion_current_line_g"]) < 3):
        previous_distortion = distortion_left
        previous_view = undistort_view_left
        st.session_state["distortion_lines_g"] = st.session_state["distortion_lines_g"] + [st.session_state["distortion_current_line_g"]]
        st.session_state["distortion_current_line_g"] = []
        st.session_state["distortion_g"] = optimize_distortion(st.session_state["distortion_lines_g"], left_frame.shape[1], left_frame.shape[0])
        st.session_state["undistort_view_g"] = compute_undistort_view_transform(left_frame.shape[1], left_frame.shape[0], st.session_state["distortion_g"])
        if st.session_state["vid_pts_g"]:
            corrected_points = invert_view_transform_to_points(st.session_state["vid_pts_g"], previous_view)
            raw_points = invert_radial_correction_to_points(corrected_points, previous_distortion)
            corrected_points = apply_radial_correction_to_points(raw_points, st.session_state["distortion_g"])
            st.session_state["vid_pts_g"] = apply_view_transform_to_points(corrected_points, st.session_state["undistort_view_g"])
        st.session_state["pending_camera"] = None
        st.session_state["pending_video_point"] = None
        st.rerun()
    if left_controls[3].button("Reset left lines"):
        if distortion_left and distortion_left.get("enabled") and st.session_state["vid_pts_g"]:
            corrected_points = invert_view_transform_to_points(st.session_state["vid_pts_g"], undistort_view_left)
            st.session_state["vid_pts_g"] = invert_radial_correction_to_points(corrected_points, distortion_left)
        st.session_state["distortion_lines_g"] = []
        st.session_state["distortion_current_line_g"] = []
        st.session_state["distortion_g"] = None
        st.session_state["undistort_view_g"] = None
        st.session_state["pending_camera"] = None
        st.session_state["pending_video_point"] = None
        st.rerun()

    st.markdown("**Right camera**")
    right_click = click_widget(right_image, 980, key="video_right")
    if right_click is not None and list(right_click) != st.session_state["prev_click_video_d"]:
        st.session_state["prev_click_video_d"] = list(right_click)
        if st.session_state["distortion_mode"]:
            if list(right_click) != st.session_state["prev_click_distortion_d"]:
                st.session_state["prev_click_distortion_d"] = list(right_click)
                st.session_state["distortion_current_line_d"] = st.session_state["distortion_current_line_d"] + [list(right_click)]
        else:
            st.session_state["pending_camera"] = "d"
            st.session_state["pending_video_point"] = list(right_click)
        st.rerun()

    right_controls = st.columns(4)
    if right_controls[0].button("Undo last right pair", disabled=len(right_video_points) == 0):
        st.session_state["vid_pts_d"] = right_video_points[:-1]
        st.session_state["ter_pts_d"] = right_terrain_points[:-1]
        st.rerun()
    if right_controls[1].button("Reset right camera"):
        st.session_state["vid_pts_d"] = []
        st.session_state["ter_pts_d"] = []
        if st.session_state["pending_camera"] == "d":
            st.session_state["pending_camera"] = None
            st.session_state["pending_video_point"] = None
        st.rerun()
    if right_controls[2].button("Validate right line", disabled=len(st.session_state["distortion_current_line_d"]) < 3):
        previous_distortion = distortion_right
        previous_view = undistort_view_right
        st.session_state["distortion_lines_d"] = st.session_state["distortion_lines_d"] + [st.session_state["distortion_current_line_d"]]
        st.session_state["distortion_current_line_d"] = []
        st.session_state["distortion_d"] = optimize_distortion(st.session_state["distortion_lines_d"], right_frame.shape[1], right_frame.shape[0])
        st.session_state["undistort_view_d"] = compute_undistort_view_transform(right_frame.shape[1], right_frame.shape[0], st.session_state["distortion_d"])
        if st.session_state["vid_pts_d"]:
            corrected_points = invert_view_transform_to_points(st.session_state["vid_pts_d"], previous_view)
            raw_points = invert_radial_correction_to_points(corrected_points, previous_distortion)
            corrected_points = apply_radial_correction_to_points(raw_points, st.session_state["distortion_d"])
            st.session_state["vid_pts_d"] = apply_view_transform_to_points(corrected_points, st.session_state["undistort_view_d"])
        st.session_state["pending_camera"] = None
        st.session_state["pending_video_point"] = None
        st.rerun()
    if right_controls[3].button("Reset right lines"):
        if distortion_right and distortion_right.get("enabled") and st.session_state["vid_pts_d"]:
            corrected_points = invert_view_transform_to_points(st.session_state["vid_pts_d"], undistort_view_right)
            st.session_state["vid_pts_d"] = invert_radial_correction_to_points(corrected_points, distortion_right)
        st.session_state["distortion_lines_d"] = []
        st.session_state["distortion_current_line_d"] = []
        st.session_state["distortion_d"] = None
        st.session_state["undistort_view_d"] = None
        st.session_state["pending_camera"] = None
        st.session_state["pending_video_point"] = None
        st.rerun()


st.divider()
preview_left, preview_right = st.columns(2, gap="large")

with preview_left:
    st.subheader("Left camera debug")
    if left_result is None:
        st.info("Add at least 4 left pairs to compute the left homography.")
    else:
        st.image(Image.fromarray(preview_projection(left_result[0], left_video_points_corrected, left_terrain_points, left_result[1], LEFT_COLOR, "L")), use_container_width=True)
        st.image(Image.fromarray(draw_video_sampling_grid(remap_frame_with_distortion(left_frame, distortion_left, undistort_view_left), left_result[0])), use_container_width=True)
        if distortion_left is not None:
            st.image(Image.fromarray(remap_frame_with_distortion(left_frame, distortion_left, undistort_view_left)), use_container_width=True)
        with st.expander("Left point errors"):
            for index, error_value in enumerate(left_result[3]):
                state = "inlier" if left_result[1][index] else "outlier"
                st.write(f"L{index + 1}: {error_value:.2f} px ({state})")

with preview_right:
    st.subheader("Right camera debug")
    if right_result is None:
        st.info("Add at least 4 right pairs to compute the right homography.")
    else:
        st.image(Image.fromarray(preview_projection(right_result[0], right_video_points_corrected, right_terrain_points, right_result[1], RIGHT_COLOR, "R")), use_container_width=True)
        st.image(Image.fromarray(draw_video_sampling_grid(remap_frame_with_distortion(right_frame, distortion_right, undistort_view_right), right_result[0])), use_container_width=True)
        if distortion_right is not None:
            st.image(Image.fromarray(remap_frame_with_distortion(right_frame, distortion_right, undistort_view_right)), use_container_width=True)
        with st.expander("Right point errors"):
            for index, error_value in enumerate(right_result[3]):
                state = "inlier" if right_result[1][index] else "outlier"
                st.write(f"R{index + 1}: {error_value:.2f} px ({state})")


st.divider()
st.subheader("Save")
st.info("Final target frame: the exact pixels of terrain.png.")

if left_result is not None and right_result is not None:
    if st.button("Save calibration.json", type="primary"):
        left_bounds = bounds_from_points(left_terrain_points) if left_terrain_points else None
        right_bounds = bounds_from_points(right_terrain_points) if right_terrain_points else None
        left_view = st.session_state.get("undistort_view_g") or {"scale": 1.0, "offset_x": 0.0, "offset_y": 0.0}
        right_view = st.session_state.get("undistort_view_d") or {"scale": 1.0, "offset_x": 0.0, "offset_y": 0.0}
        path = save_calibration(
            output_dir,
            terrain_path=terrain_path,
            terrain_width=terrain_w,
            terrain_height=terrain_h,
            left_video_points=left_video_points,
            left_terrain_points=left_terrain_points,
            left_homography=left_result[0],
            left_inliers=left_result[1],
            left_rms_error=left_result[2],
            left_point_errors=left_result[3],
            left_distortion=st.session_state.get("distortion_g"),
            left_view=left_view,
            left_distortion_lines=st.session_state.get("distortion_lines_g", []),
            left_bounds=left_bounds,
            right_video_points=right_video_points,
            right_terrain_points=right_terrain_points,
            right_homography=right_result[0],
            right_inliers=right_result[1],
            right_rms_error=right_result[2],
            right_point_errors=right_result[3],
            right_distortion=st.session_state.get("distortion_d"),
            right_view=right_view,
            right_distortion_lines=st.session_state.get("distortion_lines_d", []),
            right_bounds=right_bounds,
            split_y=compute_split_y(left_terrain_points, right_terrain_points),
        )
        st.success(f"Saved: `{path}`")
else:
    st.warning("Each camera needs at least 4 point pairs before saving.")
