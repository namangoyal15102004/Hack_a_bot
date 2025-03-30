import cv2
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from deepface.basemodels import Facenet
from deepface import DeepFace

model = Facenet.loadModel()

# -------------------------------
# 1. Load Pretrained FaceNet Model & Database
# -------------------------------
DB_FILE = "face_embeddings.pkl"  # Path to stored embeddings

# Load stored face embeddings
def load_database():
    """Load stored face embeddings from a file."""
    if os.path.exists(DB_FILE):
        return pd.read_pickle(DB_FILE)
    return None

database = load_database()

# -------------------------------
# 2. Extract Embeddings from Image
# -------------------------------
def get_embedding(face_crop):
    """Extracts a 128D embedding using FaceNet from a cropped face."""
    if face_crop is None or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
        return None
    
    face_crop = cv2.resize(face_crop, (160, 160))  # Resize for FaceNet
    face_crop = np.expand_dims(face_crop.astype("float32"), axis=0)

    return model.predict(face_crop)[0]

# -------------------------------
# 3. Recognize Face Using Database
# -------------------------------
def recognize_face(face_crop, threshold=0.5):
    """Compares the detected face with stored embeddings and returns the best match."""
    if database is None or len(database) == 0:
        print("⚠️ No stored embeddings found!")
        return "Unknown", 0

    input_embedding = get_embedding(face_crop)
    if input_embedding is None:
        return "Unknown", 0

    best_match = "Unknown"
    max_confidence = -1

    for _, row in database.iterrows():
        stored_embedding = np.array(row["embedding"]).reshape(1, -1)
        similarity = cosine_similarity([input_embedding], stored_embedding)[0][0]

        if similarity > max_confidence and similarity > threshold:
            max_confidence = similarity
            best_match = row["name"]

    return best_match, max_confidence

# -------------------------------
# 4. Start Real-Time Face Recognition with Sony AI Detection
# -------------------------------
cap = cv2.VideoCapture(1)  # Open camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------
    # Sony AI Face Detection (Replace with actual model)
    # -------------------------------
    try:
        analysis = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False)

        if isinstance(analysis, list):
            analysis = analysis[0]

        region = analysis['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        face_crop = frame[y:y+h, x:x+w]

        # Recognize the detected face
        identity, confidence = recognize_face(face_crop)

        # Draw bounding box & label
        color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"{identity} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    except Exception as e:
        pass  # Skip frame if no face is detected

    # Show live video feed
    cv2.imshow("Sony AI + FaceNet Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 5. Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
