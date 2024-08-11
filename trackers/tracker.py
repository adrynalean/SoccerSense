from ultralytics import YOLO
import supervision as SV
import pickle
import os
import sys
import cv2
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from filterpy.kalman import KalmanFilter
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = SV.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=(['x1','y1','x2','y2']))

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}}for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players"  : [],
            "referees" : [],
            "ball"     : [] 
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detections format
            detection_supervision = SV.Detections.from_ultralytics(detection)
            

            # Convert Goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox} # 1 becasue there is only 1 ball in the game
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None, outline_color=(169, 169, 169), alpha=0.5, fill=False):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        # Create an overlay
        overlay = frame.copy()
        
        # Draw the ellipse on the overlay
        cv2.ellipse(
            overlay,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=0.0 if fill else -45,
            endAngle=360.0 if fill else 235,
            color=color,
            thickness=-1 if fill else 2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            # Calculate size of the rectangle based on the ellipse width
            rectangle_width = int(width * 1.2)
            rectangle_height = int(0.35 * width)
            radius = int(0.1 * rectangle_height)

            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = y2 + int(0.2 * width)
            y2_rect = y1_rect + rectangle_height

            # Draw filled rounded rectangle
            cv2.rectangle(overlay, (x1_rect + radius, y1_rect), (x2_rect - radius, y2_rect), color, cv2.FILLED)
            cv2.rectangle(overlay, (x1_rect, y1_rect + radius), (x2_rect, y2_rect - radius), color, cv2.FILLED)
            cv2.circle(overlay, (x1_rect + radius, y1_rect + radius), radius, color, cv2.FILLED)
            cv2.circle(overlay, (x2_rect - radius, y1_rect + radius), radius, color, cv2.FILLED)
            cv2.circle(overlay, (x1_rect + radius, y2_rect - radius), radius, color, cv2.FILLED)
            cv2.circle(overlay, (x2_rect - radius, y2_rect - radius), radius, color, cv2.FILLED)

            # Calculate text size and position based on rectangle size
            text_scale = 0.5 * (rectangle_height / 20)
            text_thickness = max(1, int(text_scale * 2))
            text_size, _ = cv2.getTextSize(str(track_id), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
            text_width, text_height = text_size
            x1_text = x_center - text_width // 2
            y1_text = y1_rect + (rectangle_height + text_height) // 2

            cv2.putText(overlay,
                        f"{track_id}",
                        (int(x1_text), int(y1_text)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale,
                        (0, 0, 0),
                        text_thickness
                        )

        # Blend the overlay with the original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame

    def draw_triangle(self, frame, bbox, color=(135, 206, 235), outline_color=(0, 0, 0), alpha=0.5):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Create an overlay
        overlay = frame.copy()

        triangle_points = np.array([
            [x, y],
            [x - 5, y - 10],
            [x + 5, y - 10]
        ])
        cv2.drawContours(overlay, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(overlay, [triangle_points], 0, outline_color, 2)

        # Blend the overlay with the original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Define the coordinates and sizes for the main rectangle
        rect_width = 600
        rect_height = 70
        rect_start_x = (frame_width - rect_width) // 2
        rect_start_y = frame_height - rect_height - 30
        rect_end_x = rect_start_x + rect_width
        rect_end_y = rect_start_y + rect_height
        rect_start = (rect_start_x, rect_start_y)
        rect_end = (rect_end_x, rect_end_y)
        radius = 20

        # Draw semi-transparent light gray background with rounded edges
        overlay = frame.copy()
        bg_color = (100, 100, 100)  # Lighter gray color
        bg_alpha = 0.8

        # Draw the background rectangle with rounded edges
        cv2.rectangle(overlay, (rect_start[0], rect_start[1] + radius), (rect_end[0], rect_end[1] - radius), bg_color, -1)
        cv2.rectangle(overlay, (rect_start[0] + radius, rect_start[1]), (rect_end[0] - radius, rect_end[1]), bg_color, -1)
        cv2.ellipse(overlay, (rect_start[0] + radius, rect_start[1] + radius), (radius, radius), 180, 0, 90, bg_color, -1)
        cv2.ellipse(overlay, (rect_end[0] - radius, rect_start[1] + radius), (radius, radius), 270, 0, 90, bg_color, -1)
        cv2.ellipse(overlay, (rect_start[0] + radius, rect_end[1] - radius), (radius, radius), 90, 0, 90, bg_color, -1)
        cv2.ellipse(overlay, (rect_end[0] - radius, rect_end[1] - radius), (radius, radius), 0, 0, 90, bg_color, -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

        # Draw team titles
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        font_thickness = 2
        text_alpha = 0.5  # Opacity of the text titles

        # Create another overlay for text to adjust opacity
        text_overlay = frame.copy()
        cv2.putText(text_overlay, "Team 1", (rect_start[0] + 20, rect_start[1] + 30), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(text_overlay, "Team 2", (rect_end[0] - 100, rect_start[1] + 30), font, font_scale, (0, 0, 0), font_thickness)

        cv2.addWeighted(text_overlay, text_alpha, frame, 1 - text_alpha, 0, frame)

        # Draw possession percentages beneath the team titles
        font_scale_stat = 0.5
        font_thickness_stat = 1

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        total_frames = team1_num_frames + team2_num_frames

        if total_frames > 0:
            team1_possession = team1_num_frames / total_frames
            team2_possession = team2_num_frames / total_frames
        else:
            team1_possession = 0
            team2_possession = 0

        cv2.putText(frame, f"{team1_possession * 100:.2f}%", (rect_start[0] + 20, rect_start[1] + 60), font, font_scale_stat, (255, 255, 255), font_thickness_stat)
        cv2.putText(frame, f"{team2_possession * 100:.2f}%", (rect_end[0] - 100, rect_start[1] + 60), font, font_scale_stat, (0, 0, 0), font_thickness_stat)

        # Draw the bar graph
        bar_rect_start = (rect_start[0] + 150, rect_start[1] + 30)
        bar_rect_end = (rect_end[0] - 150, rect_end[1] - 30)
        bar_height = bar_rect_end[1] - bar_rect_start[1]
        bar_width = bar_rect_end[0] - bar_rect_start[0]

        team1_bar_width = int(bar_width * team1_possession)
        team2_bar_width = int(bar_width * team2_possession)

        bar_alpha = 0.5  # Opacity of the bar graph

        # Create another overlay for bars to adjust opacity
        bar_overlay = overlay.copy()

        # Draw Team 1 bar (left, white) with rounded edges
        if team1_bar_width > 0:
            team1_bar_end = (bar_rect_start[0] + team1_bar_width, bar_rect_end[1])
            cv2.rectangle(bar_overlay, (bar_rect_start[0], bar_rect_start[1] + radius // 2), 
                        (team1_bar_end[0], bar_rect_end[1] - radius // 2), (255, 255, 255), -1)
            cv2.rectangle(bar_overlay, (bar_rect_start[0] + radius // 2, bar_rect_start[1]), 
                        (team1_bar_end[0] - radius // 2, bar_rect_end[1]), (255, 255, 255), -1)
            cv2.ellipse(bar_overlay, (bar_rect_start[0] + radius // 2, bar_rect_start[1] + radius // 2), 
                        (radius // 2, radius // 2), 180, 0, 90, (255, 255, 255), -1)
            cv2.ellipse(bar_overlay, (team1_bar_end[0] - radius // 2, bar_rect_start[1] + radius // 2), 
                        (radius // 2, radius // 2), 270, 0, 90, (255, 255, 255), -1)
            cv2.ellipse(bar_overlay, (bar_rect_start[0] + radius // 2, bar_rect_end[1] - radius // 2), 
                        (radius // 2, radius // 2), 90, 0, 90, (255, 255, 255), -1)
            cv2.ellipse(bar_overlay, (team1_bar_end[0] - radius // 2, bar_rect_end[1] - radius // 2), 
                        (radius // 2, radius // 2), 0, 0, 90, (255, 255, 255), -1)

        # Draw Team 2 bar (right, black) with rounded edges
        if team2_bar_width > 0:
            team2_bar_start = (bar_rect_start[0] + team1_bar_width, bar_rect_start[1])
            team2_bar_end = (team2_bar_start[0] + team2_bar_width, bar_rect_end[1])
            cv2.rectangle(bar_overlay, (team2_bar_start[0], team2_bar_start[1] + radius // 2), 
                        (team2_bar_end[0], team2_bar_end[1] - radius // 2), (0, 0, 0), -1)
            cv2.rectangle(bar_overlay, (team2_bar_start[0] + radius // 2, team2_bar_start[1]), 
                        (team2_bar_end[0] - radius // 2, team2_bar_end[1]), (0, 0, 0), -1)
            cv2.ellipse(bar_overlay, (team2_bar_start[0] + radius // 2, team2_bar_start[1] + radius // 2), 
                        (radius // 2, radius // 2), 180, 0, 90, (0, 0, 0), -1)
            cv2.ellipse(bar_overlay, (team2_bar_end[0] - radius // 2, team2_bar_start[1] + radius // 2), 
                        (radius // 2, radius // 2), 270, 0, 90, (0, 0, 0), -1)
            cv2.ellipse(bar_overlay, (team2_bar_start[0] + radius // 2, team2_bar_end[1] - radius // 2), 
                        (radius // 2, radius // 2), 90, 0, 90, (0, 0, 0), -1)
            cv2.ellipse(bar_overlay, (team2_bar_end[0] - radius // 2, team2_bar_end[1] - radius // 2), 
                        (radius // 2, radius // 2), 0, 0, 90, (0, 0, 0), -1)

        cv2.addWeighted(bar_overlay, bar_alpha, frame, 1 - bar_alpha, 0, frame)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames): 
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                team_color = tuple(map(int, player["team_color"]))  # Convert color to integer tuple
                has_ball = player.get('has_ball', False)
                frame = self.draw_ellipse(frame, player["bbox"], team_color, track_id, fill=has_ball)


            # Draw referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (255, 215, 0), track_id)

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"])  # Example for the ball

            # Draw team ball control stats
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

            