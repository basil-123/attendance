import cv2
from deepface import DeepFace
from retinaface import RetinaFace
import os
import tensorflow as tf

# Check for GPU availability and configure
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU(s) available and configured")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, falling back to CPU")

# Path configurations
video_path = r"C:\Users\basil\OneDrive\Desktop\basil\deepface\videos\group.mp4"
db_path = r"C:\Users\basil\OneDrive\Desktop\basil\deepface\database"
results_file = "results.txt"
output_folder = "annotated_frames"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize RetinaFace with GPU backend
# Note: RetinaFace automatically uses GPU if available and if built with GPU support
# For OpenCV, we'll check if GPU-accelerated version is available

# Check if OpenCV is built with CUDA support
try:
    cv2.cuda.getCudaEnabledDeviceCount()
    print("OpenCV with CUDA support detected")
    # Create a CUDA-enabled video capture
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    # Create CUDA stream for async processing
    stream = cv2.cuda_Stream()
except:
    print("OpenCV without CUDA support, falling back to CPU")
    cap = cv2.VideoCapture(video_path)
    stream = None

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate of the video: {fps} FPS")

# Configure DeepFace to use GPU
# DeepFace automatically uses TensorFlow's GPU backend if available
# For explicit control, we can set backend:
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Open results file for appending
with open(results_file, "a") as file:
    frame_count = 0
    frame_skip = int(fps)  # Skip frames based on the FPS to get 1 frame per second

    while cap.isOpened() and frame_count < 10:  # Process only the first 10 frames
        for _ in range(frame_skip - 1):
            cap.grab()
        
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame or end of video reached")
            break
        
        frame_count += 1
        
        # If using CUDA, upload frame to GPU
        if stream is not None:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(img, stream=stream)
            # For RetinaFace, we still need to download for processing
            img = gpu_frame.download()
        
        # RetinaFace detection with potential GPU acceleration
        detections = RetinaFace.detect_faces(img)
        recognized_identities = set()  # Track recognized people in the same frame

        for idx, (key, face) in enumerate(detections.items(), start=1):
            x1, y1, x2, y2 = face['facial_area']
            face_img = img[y1:y2, x1:x2]
            cropped_face_path = f"temp_face_{frame_count}_{idx}.jpg"
            cv2.imwrite(cropped_face_path, face_img)

            matched_identity = "Unknown"
            best_distance = float("inf")

            try:
                # DeepFace.find() will automatically use GPU if TensorFlow is GPU-enabled
                dfs = DeepFace.find(
                    img_path=cropped_face_path,
                    db_path=db_path,
                    model_name="Facenet512",
                    distance_metric="cosine",
                    enforce_detection=False,
                    detector_backend='retinaface'  # Use retinaface for consistency
                )
            except Exception as e:
                print(f"Error processing face {idx}: {e}")
                dfs = []

            if len(dfs) > 0:
                for df in dfs:
                    for _, row in df.iterrows():
                        identity = row['identity']
                        folder_name = os.path.basename(os.path.dirname(identity))
                        distance = row['distance']
                        threshold = row['threshold']

                        if distance <= threshold and distance < best_distance and folder_name not in recognized_identities:
                            matched_identity = folder_name
                            best_distance = distance
                            recognized_identities.add(folder_name)

            # Drawing on GPU if available
            if stream is not None:
                # Download frame back to CPU for drawing (or use CUDA drawing functions if available)
                img = gpu_frame.download()
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) if matched_identity != "Unknown" else (0, 0, 255), 2)
            cv2.putText(img, matched_identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if matched_identity != "Unknown" else (0, 0, 255), 2)

            if stream is not None:
                gpu_frame.upload(img, stream=stream)
            
            with open(results_file, "a") as file:
                file.write(f"Frame {frame_count}, Face {idx}: {matched_identity} (Bounding Box: x={x1}, y={y1}, w={x2-x1}, h={y2-y1})\n")
            
            print(f"Frame {frame_count}, Face {idx}: {matched_identity} (Bounding Box: x={x1}, y={y1}, w={x2-x1}, h={y2-y1})")
            os.remove(cropped_face_path)
        
        annotated_image_path = os.path.join(output_folder, f"annotated_frame_{frame_count}.jpg")
        cv2.imwrite(annotated_image_path, img)  # Save annotated frame

        cv2.imshow("Video Face Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print(f"Results saved to: {results_file}")
print(f"Annotated images saved in: {output_folder}")