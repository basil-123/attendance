import cv2
from . import Model
from retinaface import RetinaFace
import os
from datetime import datetime

# Path to your database where images of known people are stored
db_path = r"C:\Users\basil\OneDrive\Desktop\basil\deepface\database"
results_file = "results.txt"  # File to save results

# Initialize webcam (use 0 for the default webcam)
cap = cv2.VideoCapture(0)

# Track already matched identities
matched_identities = set()

# Open results file for appending
with open(results_file, "a") as file:
    while True:
        # Read a frame from the webcam
        ret, img = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Detect faces using RetinaFace
        detections = RetinaFace.detect_faces(img)

        # Iterate over each detected face
        for idx, (key, face) in enumerate(detections.items(), start=1):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = face['facial_area']
            face_img = img[y1:y2, x1:x2]  # Crop the face

            # Save the cropped face temporarily
            cropped_face_path = f"temp_face_{idx}.jpg"
            cv2.imwrite(cropped_face_path, face_img)

            try:
                # Perform face recognition on the cropped face
                dfs = Model.find(
                    img_path=cropped_face_path,
                    db_path=db_path,
                    model_name="VGG-Face",  # Choose your model
                    distance_metric="cosine",  # Set distance metric
                    enforce_detection=False  # Avoid errors if no face is found
                )
            except Exception as e:
                print(f"Error processing face {idx}: {e}")
                dfs = []

            # Initialize the label
            label = "Unknown"
            best_distance = float("inf")

            # Analyze results for the current face
            if len(dfs) > 0:
                for df in dfs:
                    for _, row in df.iterrows():
                        identity = row['identity']
                        folder_name = os.path.basename(os.path.dirname(identity))
                        distance = row['distance']
                        threshold = row['threshold']

                        # Skip already matched identities
                        if folder_name in matched_identities:
                            continue

                        # Update label if the distance is within the threshold
                        if distance <= threshold and distance < best_distance:
                            label = folder_name
                            best_distance = distance

            # Mark the current identity as matched if it is not "Unknown"
            if label != "Unknown":
                matched_identities.add(label)

            # Annotate the face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)

            # Write result to the text file
            with open(results_file, "a") as file:
                file.write(f"Face {idx}: {label} (Bounding Box: x={x1}, y={y1}, w={x2-x1}, h={y2-y1})\n")
            
            # Print the result in terminal
            print(f"Face {idx}: {label} (Bounding Box: x={x1}, y={y1}, w={x2-x1}, h={y2-y1})")

            # Remove the temporary file
            os.remove(cropped_face_path)

        # Display the annotated frame
        cv2.imshow("Webcam Face Recognition", img)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

print(f"Results saved to: {results_file}")
