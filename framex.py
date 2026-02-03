import cv2
import os

# Input video path
video_path = r"C:\Users\basil\OneDrive\Desktop\basil\deepface\videos\video.mp4"  # Replace with your video file path

# Folder to store the extracted frames
output_folder = 'frames'  # Folder to save the extracted frames
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the FPS (frames per second) of the video
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video
print(f"FPS of video: {fps}")

# Define the frame extraction rate (e.g., 1 frame per second)
frame_interval = int(fps)  # Extract 1 frame per second

frame_count = 0
current_frame = 0

while cap.isOpened():
    ret, frame = cap.read()  # Read each frame
    if not ret:
        break  # Exit if no more frames are available

    # Save the frame if the current frame is a multiple of frame_interval
    if current_frame % frame_interval == 0:
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)  # Save the frame as an image
        print(f"Saved: {frame_filename}")
        frame_count += 1

    current_frame += 1

# Release the video capture object and finalize
cap.release()
print(f"Extracted {frame_count} frames to {output_folder}")
