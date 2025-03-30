import sys
import time
import cv2
import torch
import pickle
import numpy as np
import re
import pytesseract
from pyzbar.pyzbar import decode
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms
from picamera2 import MappedArray, Picamera2, Preview
from PyQt5.QtWidgets import QApplication

##############################################################################
# UTILITY FUNCTIONS
##############################################################################

def beep():
    """
    Simple 'beep' placeholder that just prints an alert character.
    You could replace with a real beep or buzzer on the Pi.
    """
    print('\a')  # Terminal beep (may or may not be audible depending on system)


def extract_id_number(frame):
    """
    Extracts an ID number from a frame using OCR.
    Example: tries to match 'ID Number: 1234567' or any 7+ digits.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Simple threshold to help Tesseract
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh)

        # Look for a string like "ID Number: 1234..."
        id_match = re.search(r'ID\s*Number:?\s*(\d+)', text, re.IGNORECASE)
        if id_match:
            return id_match.group(1)

        # Otherwise, look for any 7+ digit chunk
        digits_match = re.search(r'(\d{7,})', text)
        if digits_match:
            return digits_match.group(1)

        return None
    except Exception as e:
        print(f"OCR error: {e}")
        return None


##############################################################################
# OVERLAY DRAWING: for barcode + OCR
##############################################################################
def draw_overlay(request):
    """
    Draws overlay for barcodes and OCR text in the preview.
    This will be invoked automatically by Picamera2 for each displayed frame.
    """
    global barcodes, display_data, ocr_mode, start_time

    colour_barcode = (0, 255, 0)
    colour_ocr = (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    with MappedArray(request, "main") as m:
        current_time = time.time()
        elapsed_time = current_time - start_time
        status_text = "Scanning barcodes..." if not ocr_mode else "OCR MODE ACTIVE"
        cv2.putText(m.array, status_text, (10, 30), font, 0.7, (0, 0, 255), 2)

        # Draw detected barcodes
        for b in barcodes:
            if b.polygon:
                points = np.array([(p.x, p.y) for p in b.polygon])
                cv2.polylines(m.array, [points], True, colour_barcode, 2)
                x, y = min(p.x for p in b.polygon), min(p.y for p in b.polygon) - 30
                cv2.putText(m.array, b.data.decode('utf-8'), (x, y), font, 0.7, colour_barcode, 2)

        # Draw OCR text boxes (if in OCR mode)
        if ocr_mode:
            for item in display_data:
                x, y, w, h = item["box"]
                cv2.putText(m.array, item["text"], (x, y), font, 0.7, colour_ocr, 2)


##############################################################################
# BARCODE/QR SCANNER + OCR LOGIC
##############################################################################
def run_scanner(picam2):
    """
    Continuously scans for barcodes using Pyzbar. If no barcode is found for 30s,
    fall back to OCR mode to extract an ID. Writes found results to scanned_barcodes.txt.
    Press Ctrl+C to stop or return from this function.
    """
    global barcodes, display_data, ocr_mode, start_time

    print("Starting barcode/QR scanning with OCR fallback...")
    # Initialize states:
    start_time = time.time()
    barcodes = []
    display_data = []
    ocr_mode = False

    detected_barcodes = {}
    cooldown_time = 2    # Time to wait before the same barcode can trigger again
    ocr_threshold = 50   # Confidence threshold for Tesseract recognized text
    ocr_cooldown = 5     # Time to wait between OCR attempts
    last_ocr_time = 0

    # Attach the overlay callback so we can draw bounding boxes
    picam2.post_callback = draw_overlay

    try:
        while True:
            # Grab current frame as a numpy array
            frame = picam2.capture_array("main")
            current_time = time.time()
            elapsed_time = current_time - start_time

            # If 30s has passed without detecting any barcode => switch to OCR mode
            if not ocr_mode and elapsed_time > 30:
                print("No barcode detected for 30 seconds. Activating OCR mode.")
                ocr_mode = True

            # Attempt to decode barcodes every frame
            barcode_detected = False
            barcodes = decode(frame)
            if barcodes:
                for barcode in barcodes:
                    barcode_data = barcode.data.decode('utf-8')
                    # Only trigger if it's a new barcode or cooldown has passed
                    if (barcode_data not in detected_barcodes or
                       (current_time - detected_barcodes[barcode_data] > cooldown_time)):

                        print(f"Time: {time.strftime('%H:%M:%S')}")
                        print(f"Barcode: {barcode_data}")
                        print("-------------------")

                        # Save timestamp + barcode to file
                        with open("scanned_barcodes.txt", "a") as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {barcode_data}\n")

                        beep()

                        detected_barcodes[barcode_data] = current_time
                        start_time = current_time   # Reset the no-barcode timer
                        ocr_mode = False            # Found a barcode => turn off OCR
                        barcode_detected = True

            # If in OCR mode and still no barcode => do an OCR attempt every ocr_cooldown
            if ocr_mode and not barcode_detected and (current_time - last_ocr_time > ocr_cooldown):
                print("Attempting OCR to extract ID number...")
                ocr_data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
                extracted_data = []

                # Collect recognized text with confidence > ocr_threshold
                for i in range(len(ocr_data['text'])):
                    text_val = ocr_data['text'][i].strip()
                    if text_val:  # Non-empty
                        conf = ocr_data['conf'][i]
                        # Tesseract can sometimes return conf as a string, so convert to int if possible
                        if isinstance(conf, str) and conf.isdigit():
                            conf = int(conf)

                        if conf > ocr_threshold:
                            extracted_data.append({
                                "text": text_val,
                                "conf": conf,
                                "box": (ocr_data['left'][i], ocr_data['top'][i],
                                        ocr_data['width'][i], ocr_data['height'][i])
                            })

                # Attempt to parse out an ID from the recognized text
                id_number = extract_id_number(frame)
                if id_number:
                    print(f"ID Number (via OCR): {id_number}")
                    with open("scanned_barcodes.txt", "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ID Number: {id_number}\n")
                    beep()

                display_data = extracted_data
                last_ocr_time = current_time

            # Limit CPU usage a bit
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nUser pressed Ctrl+C. Exiting scanner...")


##############################################################################
# FACE RECOGNITION LOGIC
##############################################################################
def run_face_recognition(picam2, svm_classifier, resnet, mtcnn, threshold=0.70):
    """
    Continuously captures frames, detects a face, extracts embeddings, and uses
    the SVM to label it. If no face is found or face is unknown for 30s, sets a
    'k=1' flag to break and do something else (like scanning).
    Press 'q' to quit face recognition manually.
    """
    to_tensor = transforms.ToTensor()
    starttime = time.time()

    while True:
        # Capture frame from Picamera2 (RGB)
        frame_rgb = picam2.capture_array()[:, :, :3]
        frame_bgr = frame_rgb.copy()  # for OpenCV display (BGR)

        # Detect the single largest face
        boxes, probs = mtcnn.detect(frame_rgb)
        if boxes is None:
            cv2.putText(frame_bgr, "No face detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            elapsed = time.time() - starttime
            if elapsed >= 30:
                # 30s with no face => let's break out
                print("No face detected for 30s; switching to scanning mode.")
                break
        else:
            # We got at least one face box
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]  # largest face
            # clip region to valid area
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame_rgb.shape[1]), min(y2, frame_rgb.shape[0])
            face_crop = frame_rgb[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            face_resized = cv2.resize(face_crop, (160, 160))
            face_tensor = to_tensor(face_resized).unsqueeze(0)

            # Generate Face Embedding
            with torch.no_grad():
                embedding = resnet(face_tensor).cpu().numpy()

            # Predict with SVM
            probabilities = svm_classifier.predict_proba(embedding)[0]
            max_prob = np.max(probabilities)
            predicted_label = svm_classifier.classes_[np.argmax(probabilities)]

            if max_prob >= threshold:
                label = predicted_label
                # Reset timer if face is recognized
                starttime = time.time()
            else:
                label = "Unknown"
                # If 'Unknown' for 30s => switch to scanning
                elapsed = time.time() - starttime
                if elapsed >= 30:
                    print("Unknown face for 30s; switching to scanning mode.")
                    break

            # Draw bounding box & label
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_bgr, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show face recognition window
        cv2.imshow("Live Face Recognition", frame_bgr)

        # If user presses 'q', break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("User pressed 'q' in face window. Exiting face recognition.")
            break

    # Destroy the face recognition OpenCV window on exit
    cv2.destroyWindow("Live Face Recognition")


##############################################################################
# MAIN ENTRY POINT
##############################################################################
def main():
    """
    1) Create a QApplication (required for Qt + Picamera2).
    2) Load the SVM and set up face recognition pipeline (MTCNN + InceptionResnetV1).
    3) Start a single Picamera2 instance and Qt preview (once).
    4) Run face recognition until no face or unknown face for 30s, or user quits.
    5) Switch to barcode/QR scanning with OCR fallback.
    """
    app = QApplication(sys.argv)  # Must be in main thread for Qt

    # ---------------------------
    # 1. Load Pre-trained SVM
    # ---------------------------
    with open('svm_classifier.pkl', 'rb') as f:
        svm_classifier = pickle.load(f)

    # ---------------------------
    # 2. MTCNN + InceptionResnet
    # ---------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # ---------------------------
    # 3. Initialize Picamera2 (once), and start preview with Qt
    # ---------------------------
    picam2 = Picamera2()

    # Use a lower resolution for face recognition for speed
    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (320, 240)}
    )
    picam2.configure(config)

    # Start Qt-based preview exactly once
    picam2.start_preview(Preview.QTGL)
    picam2.start()

    # ---------------------------
    # 4. Face Recognition Phase
    # ---------------------------
    print("=== Starting face recognition mode ===")
    run_face_recognition(picam2, svm_classifier, resnet, mtcnn)

    # ---------------------------
    # 5. Switch to Scanning Phase
    # ---------------------------
    print("=== Switching to Barcode/OCR Scanning ===")
    run_scanner(picam2)

    # Keep Qt running until user closes it or Ctrl+C
    sys.exit(app.exec_())


# Standard Python guard for the main function
if __name__ == "__main__":
    main()