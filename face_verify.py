import cv2
from deepface import DeepFace
import numpy as np

# -------------------------------
# 1. Specify the target image
# -------------------------------
# This is the reference image (the face you want to verify against).
target_image_path = "/Users/user/Desktop/5719E1D7-E850-4A4E-AC02-2504B5D5DF21_1_105_c.jpeg"  # <-- Change this path to your target face image

# -------------------------------
# 2. Open the Video Capture
# -------------------------------
cap = cv2.VideoCapture(1)  # 0 opens the default webcam

# -------------------------------
# 3. Process Video Frames in a Loop
# -------------------------------
while True:
    ret, frame = cap.read()  # Read one frame from the video feed
    if not ret:
        break  # If no frame is returned, exit the loop

    # Optionally, resize the frame for faster processing:
    # frame = cv2.resize(frame, (640, 480))

    # -------------------------------
    # 4. Face Detection and Analysis
    # -------------------------------
    try:
        # DeepFace.analyze() will detect faces and return details such as bounding box ("region")
        # We request 'age' and 'gender' just as a placeholder; the primary goal is to get the facial region.
        analysis = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False)
        
        # If multiple faces are detected, DeepFace.analyze() returns a list.
        # For simplicity, we will use the first detected face.
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        # Extract the bounding box of the detected face.
        region = analysis['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        
        # Crop the face from the frame using the detected coordinates.
        face_crop = frame[y:y+h, x:x+w]
        
        # -------------------------------
        # 5. Face Verification
        # -------------------------------
        # Compare the target face (from target_image_path) with the cropped face from the video.
        # Set enforce_detection=False to skip running a new detection on the cropped image.
        verification_result = DeepFace.verify(target_image_path, face_crop, enforce_detection=False)
        
        # Decide on a label based on the verification result.
        label = "Verified" if verification_result["verified"] else "Not Verified"
        
        # -------------------------------
        # 6. Draw the Results on the Frame
        # -------------------------------
        # Draw a rectangle around the detected face.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Put a label above the rectangle.
        color = (0, 255, 0) if verification_result["verified"] else (0, 0, 255)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    except Exception as e:
        # If an error occurs (e.g., no face detected in the frame), print the error for debugging
        # or simply pass to ignore and process the next frame.
        # Uncomment the next line to see error messages:
        # print("Error:", e)
        pass

    # -------------------------------
    # 7. Display the Video Frame
    # -------------------------------
    cv2.imshow("Live Face Verification", frame)
    
    # Break out of the loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 8. Cleanup: Release the Camera and Close Windows
# -------------------------------
cap.release()
cv2.destroyAllWindows()
