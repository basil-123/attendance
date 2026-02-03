from . import Model
from retinaface import RetinaFace
import cv2
import os

# Paths
img_path = r"C:\Users\basil\OneDrive\Desktop\basil\deepface\group\group3.jpg"
db_path = r"C:\Users\basil\OneDrive\Desktop\basil\deepface\database"

# Load the group photo
img = cv2.imread(img_path)

# Detect faces using RetinaFace
detections = RetinaFace.detect_faces(img)

# Iterate over each detected face
print("Face Recognition Results:")
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

                # Update label if the distance is within the threshold
                if distance <= threshold and distance < best_distance:
                    label = folder_name
                    best_distance = distance

    # Annotate the face
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)

    # Print the result for the face
    print(f"Face {idx}: Label: {label}, Bounding Box: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")

    # Remove the temporary file
    os.remove(cropped_face_path)

# Save and show the annotated image
output_path = "annotated_image.jpg"
cv2.imwrite(output_path, img)
cv2.imshow("Annotated Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Annotated image saved to: {output_path}")
