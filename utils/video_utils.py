# Will have the utility to read and save the video
import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        # The ret is a flag to tell whether it is actually a frame or if the video ahs ended
        ret, frame = cap.read()
        if not ret:
            # The video has ended
            break
        frames.append(frame)
    
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()
