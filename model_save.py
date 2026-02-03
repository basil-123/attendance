import os
from deepface.models.facial_recognition.Facenet import InceptionResNetV1
from deepface.commons import weight_utils

# 1. Create a custom loader that bypasses the automatic saving
def safe_load_facenet512d():
    # Create model architecture
    model = InceptionResNetV1(dimension=512)
    
    # Download weights
    weight_file = weight_utils.download_weights_if_necessary(
        file_name="facenet512_weights.h5",
        source_url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5"
    )
    
    # Load weights without automatic saving
    model.load_weights(weight_file)
    return model

# 2. Create output directory
os.makedirs("saved_models", exist_ok=True)

try:
    # 3. Load model safely
    model = safe_load_facenet512d()
    
    # 4. Save with proper extension (choose one method)
    
    # Option 1: Save as .keras (recommended)
    model.save("saved_models/model.h5")
    print("Model saved successfully as .keras format")
    
    # Option 2: Save as .h5
    # model.save("saved_models/facenet512d.h5")
    # print("Model saved successfully as .h5 format")
    
    # Option 3: Save as SavedModel
    # model.export("saved_models/facenet512d_savedmodel")
    # print("Model saved successfully in SavedModel format")

except Exception as e:
    print(f"Error saving model: {str(e)}")