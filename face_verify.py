import cv2
import torch
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms
import time
import pytesseract
import re
from pyzbar.pyzbar import decode
from picamera2 import MappedArray, Picamera2, Preview

# Initialize Picamera2 globally
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (320, 240)})
picam2.configure(config)

# Initialize variables
k = 0
start_time = 0
barcodes = []
display_data = []
ocr_mode = False
face_mode = True  # Start in face mode
preview_running = False  # Track the state of preview

# 1. Load Pre-trained SVM for face classification
with open('svm_classifier.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN and InceptionResnetV1 for face detection and recognition
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Threshold for unknown face recognition
THRESHOLD = 0.70

# Preload transformations
to_tensor = transforms.ToTensor()

def beep():
    print('\a')  # Print alert character

def extract_id_number(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh)
        id_match = re.search(r'ID\s*Number:?\s*(\d+)', text, re.IGNORECASE)
        if id_match:
            return id_match.group(1)
        digits_match = re.search(r'(\d{7,})', text)
        if digits_match:
            return digits_match.group(1)
        return None
    except Exception as e:
        print(f"OCR error: {e}")
        return None

def draw_overlay(request):
    global display_data, barcodes, ocr_mode
    colour_barcode = (0, 255, 0)
    colour_ocr = (0, 255, 255)
    colour_face = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    with MappedArray(request, "main") as m:
        current_time = time.time()
        elapsed_time = current_time - start_time
        status_text = "Scanning barcodes..." if not ocr_mode else "OCR MODE ACTIVE"
        if face_mode:
            status_text = "Face Recognition Mode"
        cv2.putText(m.array, status_text, (10, 30), font, 0.7, (0, 0, 255), 2)
        # Draw barcode boxes
        for b in barcodes:
            if b.polygon:
                points = np.array([(p.x, p.y) for p in b.polygon])
                cv2.polylines(m.array, [points], True, colour_barcode, 2)
                x, y = min(p.x for p in b.polygon), min(p.y for p in b.polygon) - 30
                cv2.putText(m.array, b.data.decode('utf-8'), (x, y), font, 0.7, colour_barcode, 2)
        # Draw OCR boxes
        if ocr_mode:
            for item in display_data:
                x, y, w, h = item["box"]
                cv2.putText(m.array, item["text"], (x, y), font, 0.7, colour_ocr, 2)

        # Draw face boxes and labels
        if box is not None:
            x1, y1, x2, y2 = [int(b) for b in box[0]]
            label = "Unknown"  # Default label
            if max_prob >= THRESHOLD:
                label = predicted_label
            cv2.rectangle(m.array, (x1, y1), (x2, y2), colour_face, 2)
            cv2.putText(m.array, label, (x1, y1 - 10), font, 0.7, colour_face, 2)

def main():
    global start_time, barcodes, display_data, ocr_mode, face_mode, preview_running
    print("Starting Raspberry Pi system with Facial Recognition, Barcode, and OCR...")

    # Check if preview is already running, if not, start it
    if not preview_running:
        picam2.start_preview(Preview.QTGL)
        preview_running = True

    picam2.post_callback = draw_overlay
    start_time = time.time()
    detected_barcodes = {}
    cooldown_time = 2
    ocr_threshold = 50
    ocr_cooldown = 5
    last_ocr_time = 0

    try:
        while True:
            frame = picam2.capture_array("main")
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Switch to barcode and OCR mode if no face detected for 30 seconds
            if face_mode and elapsed_time > 30:
                print("No face detected for 30 seconds. Switching to Barcode/OCR mode.")
                face_mode = False
                ocr_mode = False  # Start with barcode scanning first

            # Face detection and recognition
            if face_mode:
                box, prob = mtcnn.detect(frame)

                if box is None:
                    cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    if elapsed_time >= 30:
                        face_mode = False
                        start_time = current_time
                else:
                    x1, y1, x2, y2 = [int(b) for b in box[0]]  # Only process the first (largest) face

                    # Ensure valid coordinates
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, frame.shape[1]), min(y2, frame.shape[0])

                    # Crop face region safely
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:  # Prevents OpenCV resize error
                        continue

                    # Resize face to 160x160
                    face_resized = cv2.resize(face_crop, (160, 160))

                    # Convert to Tensor & Move to Device
                    face_tensor = to_tensor(face_resized).unsqueeze(0).to(device)

                    # Generate Face Embedding
                    with torch.no_grad():
                        embedding = resnet(face_tensor).cpu().numpy()

                    # Predict with SVM
                    probabilities = svm_classifier.predict_proba(embedding)[0]
                    max_prob = np.max(probabilities)
                    predicted_label = svm_classifier.classes_[np.argmax(probabilities)]

                    if max_prob >= THRESHOLD:
                        label = predicted_label
                    else:
                        label = "Unknown"
                        if elapsed_time >= 30:
                            face_mode = False
                            start_time = current_time

                    # Draw bounding box & label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Barcode and OCR detection
            if not face_mode:
                barcodes = decode(frame)
                barcode_detected = False

                if barcodes:
                    for barcode in barcodes:
                        barcode_data = barcode.data.decode('utf-8')
                        if barcode_data not in detected_barcodes or (current_time - detected_barcodes[barcode_data] > cooldown_time):
                            print(f"Barcode detected: {barcode_data}")
                            detected_barcodes[barcode_data] = current_time
                            with open("scanned_barcodes.txt", "a") as f:
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {barcode_data}\n")
                            beep()
                            start_time = current_time
                            barcode_detected = True

                # OCR fallback for ID numbers
                if not barcode_detected and (current_time - last_ocr_time > ocr_cooldown):
                    print("Attempting OCR to extract ID number...")
                    ocr_data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
                    extracted_data = []
                    for i in range(len(ocr_data['text'])):
                        if ocr_data['text'][i].strip():
                            conf = ocr_data['conf'][i]
                            if isinstance(conf, str) and conf.isdigit():
                                conf = int(conf)

                            if conf > ocr_threshold:
                                extracted_data.append({
                                    "text": ocr_data['text'][i],
                                    "conf": conf,
                                    "box": (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                                })

                    id_number = extract_id_number(frame)
                    if id_number:
                        print(f"ID Number (via OCR): {id_number}")
                        with open("scanned_barcodes.txt", "a") as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ID Number: {id_number}\n")
                        beep()
                    display_data = extracted_data
                    last_ocr_time = current_time

                if elapsed_time >= 30:  # Switch back to face mode after 30 seconds of no activity
                    print("No activity for 30 seconds. Switching to Face Recognition mode.")
                    face_mode = True
                    start_time = current_time

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        picam2.stop_preview()  # Ensure preview is stopped when done
        picam2.stop()
        print("System closed")

# Run the main function to start the system
main()
