import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

        self.x_movements = []
        self.y_movements = []

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def smooth_movements(self, movements, window_size=5):
        smoothed_movements = []
        for i in range(len(movements)):
            if i < window_size:
                smoothed_movements.append(movements[i])
            else:
                smoothed_movements.append(np.mean(movements[i-window_size:i], axis=0))
        return smoothed_movements

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        # Apply smoothing to camera movements
        smoothed_movements = self.smooth_movements(camera_movement_per_frame)

        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            # Step 1: Draw dark gray rounded rectangle
            rect_start = (10, 10)
            rect_end = (450, 80)  # Longer rectangle
            radius = 15
            bg_color = (100, 100, 100)  # Dark gray
            bg_alpha = 0.8

            # Draw the background rectangle with rounded edges
            overlay = frame.copy()
            cv2.rectangle(overlay, (rect_start[0], rect_start[1] + radius), (rect_end[0], rect_end[1] - radius), bg_color, -1)
            cv2.rectangle(overlay, (rect_start[0] + radius, rect_start[1]), (rect_end[0] - radius, rect_end[1]), bg_color, -1)
            cv2.ellipse(overlay, (rect_start[0] + radius, rect_start[1] + radius), (radius, radius), 180, 0, 90, bg_color, -1)
            cv2.ellipse(overlay, (rect_end[0] - radius, rect_start[1] + radius), (radius, radius), 270, 0, 90, bg_color, -1)
            cv2.ellipse(overlay, (rect_start[0] + radius, rect_end[1] - radius), (radius, radius), 90, 0, 90, bg_color, -1)
            cv2.ellipse(overlay, (rect_end[0] - radius, rect_end[1] - radius), (radius, radius), 0, 0, 90, bg_color, -1)
            cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

            # Divide the rectangle into sections
            section_height = (rect_end[1] - rect_start[1]) // 2
            section_width = (rect_end[0] - rect_start[0]) // 2

            # Get smoothed camera movements
            x_movement, y_movement = smoothed_movements[frame_num]

            # Draw labels
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5
            font_thickness = 1

            cv2.putText(frame, f"X: {x_movement:.2f}", (rect_start[0] + 20, rect_start[1] + section_height // 2), font, font_scale, (255, 255, 255), font_thickness)
            cv2.putText(frame, f"Y: {y_movement:.2f}", (rect_start[0] + 20, rect_start[1] + 3 * section_height // 2), font, font_scale, (255, 255, 255), font_thickness)

            # Draw X movement bar
            bar_overlay = frame.copy()
            bar_alpha = 0.5
            bar_x_start = rect_start[0] + 250  # Position for X bar start
            bar_x_length = int(x_movement * 6)  # Scaling (max 80 units)
            bar_x_length = max(-180 // 2, min(bar_x_length, 180 // 2))  # Ensure the bar length stays within the section
            bar_x_color = (255, 255, 255)  # White

            if bar_x_length > 0:
                cv2.rectangle(bar_overlay, (bar_x_start, rect_start[1] + section_height // 2 - 5), (bar_x_start + bar_x_length, rect_start[1] + section_height // 2 + 5), bar_x_color, -1)
            else:
                cv2.rectangle(bar_overlay, (bar_x_start + bar_x_length, rect_start[1] + section_height // 2 - 5), (bar_x_start, rect_start[1] + section_height // 2 + 5), bar_x_color, -1)

            # Draw Y movement bar
            bar_y_start = rect_start[0] + 250  # Position for Y bar start
            bar_y_length = int(y_movement * 6)  # Scaling (max 80 units)
            bar_y_length = max(-180 // 2, min(bar_y_length, 180 // 2))  # Ensure the bar length stays within the section
            bar_y_color = (0, 0, 0)  # Black

            if bar_y_length > 0:
                cv2.rectangle(bar_overlay, (bar_y_start, rect_start[1] + 3 * section_height // 2 - 5), (bar_y_start + bar_y_length, rect_start[1] + 3 * section_height // 2 + 5), bar_y_color, -1)
            else:
                cv2.rectangle(bar_overlay, (bar_y_start + bar_y_length, rect_start[1] + 3 * section_height // 2 - 5), (bar_y_start, rect_start[1] + 3 * section_height // 2 + 5), bar_y_color, -1)

            cv2.addWeighted(bar_overlay, bar_alpha, frame, 1 - bar_alpha, 0, frame)

            output_frames.append(frame)

        return output_frames
