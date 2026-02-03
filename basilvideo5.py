import time
import cv2
from deepface import DeepFace
from retinaface import RetinaFace
import os
import datetime
from supabase import create_client, Client
import uuid

# Configuration
output_video_path_first = "temp_capture_first.mp4"  # First capture
output_video_path_last = "temp_capture_last.mp4"    # Last capture
db_path = r"C:\Users\basil\OneDrive\Desktop\basil\attendance\database"
results_file = "attendance.txt"
output_folder = "annotated_frames"
capture_duration = 10  # Seconds per capture

# Supabase Configuration
#SUPABASE_URL = "https://tgikvbcjmipmjhkuyhro.supabase.co"
SUPABASE_URL = "https://gtbogxalszpurhpwidnh.supabase.co"
#SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRnaWt2YmNqbWlwbWpoa3V5aHJvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzczOTM3OTcsImV4cCI6MjA1Mjk2OTc5N30.G47NMiR6QsQHfIJjajCPkPjrr4Omg14hoS78OkhbPIs"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd0Ym9neGFsc3pwdXJocHdpZG5oIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAxMDg0MDgsImV4cCI6MjA4NTY4NDQwOH0.rebVENTEShl0Mh_drYahprBB1VwIgmKSMrcVVhzFBZE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

os.makedirs(output_folder, exist_ok=True)

def capture_video(output_path):
    """Capture video from webcam for specified duration"""
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Capturing video to {output_path}...")
    start_time = datetime.datetime.now()
    
    while (datetime.datetime.now() - start_time).seconds < capture_duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Recording (Press Q to stop early)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {output_path}")

def process_video(video_path, capture_type):
    """Process video with consistent face naming"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps)
    frame_count = 0
    
    # Create subfolder for this capture
    capture_folder = os.path.join(output_folder, capture_type)
    os.makedirs(capture_folder, exist_ok=True)
    
    # Track name assignments across frames
    face_name_map = {}  # {face_id: name}
    used_names = set()  # Names already assigned in current session
    
    while cap.isOpened():
        for _ in range(frame_skip - 1):
            cap.grab()
        
        ret, img = cap.read()
        if not ret:
            break
        
        frame_count += 1
        detections = RetinaFace.detect_faces(img) or {}
        current_frame_names = set()  # Names used in this frame
        
        # First pass: Try to recognize known faces
        for face_id, face in detections.items():
            x1, y1, x2, y2 = face['facial_area']
            face_img = img[y1:y2, x1:x2]
            temp_path = f"temp_face_{frame_count}_{face_id}.jpg"
            cv2.imwrite(temp_path, face_img)

            # Check if we've seen this face before
            if face_id in face_name_map:
                name = face_name_map[face_id]
                if name not in current_frame_names:
                    current_frame_names.add(name)
                    color = (0, 255, 0)  # Green for recognized
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, name, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                continue
            
            # New face - try to recognize
            try:
                dfs = DeepFace.find(
                    img_path=temp_path,
                    db_path=db_path,
                    model_name="Facenet512",
                    distance_metric="cosine",
                    enforce_detection=False
                )
                
                best_match = None
                best_distance = float('inf')
                
                for df in dfs:
                    if not df.empty:
                        row = df.iloc[0]
                        candidate = os.path.basename(os.path.dirname(row['identity']))
                        distance = row['distance']
                        
                        # Only consider if below threshold and name not used in this frame
                        if (distance <= row['threshold'] and 
                            distance < best_distance and 
                            candidate not in current_frame_names and
                            candidate not in used_names):
                            
                            best_match = candidate
                            best_distance = distance
                
                if best_match:
                    face_name_map[face_id] = best_match
                    current_frame_names.add(best_match)
                    used_names.add(best_match)
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, best_match, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    color = (0, 0, 255)  # Red for unknown
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, "Unknown", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            except Exception as e:
                print(f"Error processing face {face_id}: {e}")
                color = (0, 0, 255)  # Red for errors
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, "Error", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            os.remove(temp_path)
        
        # Display and save
        cv2.imshow(f"Processing {capture_type}", img)
        annotated_path = os.path.join(capture_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(annotated_path, img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return list(used_names)  # Return all unique names found

def upload_to_supabase(first10_names, last10_names):
    """Upload with unique name validation"""
    # Get intersection without converting to set to maintain order
    real_names = [name for name in first10_names if name in last10_names]
    
    record = {
        "date": datetime.datetime.now().isoformat(),
        "period": "1",
        "Subject": "maths",
        "First10": ", ".join(first10_names) if first10_names else "None",
        "last10": ", ".join(last10_names) if last10_names else "None",
        "real": ", ".join(real_names) if real_names else "None"
    }
    
    response = supabase.table("attendance1").insert(record).execute()
    return not hasattr(response, 'error') or not response.error

def main():
    # First capture (for First10)
    print("Starting FIRST 10-second capture...")
    capture_video(output_video_path_first)
    first10_attendance = process_video(output_video_path_first, "First10")
    
    # Wait 10 seconds before last10 capture
    print("\nWaiting 10 seconds before last10 capture...")
    for i in range(10, 0, -1):
        print(f"Starting last10 capture in {i} seconds...", end='\r')
        time.sleep(1)
    print("\nStarting SECOND 10-second capture...")
    
    # Second capture (for Last10)
    capture_video(output_video_path_last)
    last10_attendance = process_video(output_video_path_last, "Last10")
    
    # Calculate real attendance just for this session
    real_attendance = list(set(first10_attendance) & set(last10_attendance))
    
    # Save results
    with open(results_file, "w") as f:
        f.write(f"Attendance Record - {datetime.datetime.now()}\n")
        f.write("First 10 seconds:\n")
        f.writelines(f"- {s}\n" for s in first10_attendance)
        f.write("\nLast 10 seconds (after 10 sec delay):\n")
        f.writelines(f"- {s}\n" for s in last10_attendance)
        f.write("\nPresent in both (real):\n")
        f.writelines(f"- {s}\n" for s in real_attendance)
    
    # Upload to Supabase
    if upload_to_supabase(first10_attendance, last10_attendance):
        print("Data successfully uploaded to Supabase")
    else:
        print("Upload failed")
    
    # Cleanup
    for path in [output_video_path_first, output_video_path_last]:
        if os.path.exists(path):
            os.remove(path)
    
    print(f"\nAnnotated images saved in: {output_folder}")

if __name__ == "__main__":
    main()