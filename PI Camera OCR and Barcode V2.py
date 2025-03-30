import cv2
import time
import numpy as np
import pytesseract
import re
from pyzbar.pyzbar import decode
from picamera2 import MappedArray, Picamera2, Preview

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
    font = cv2.FONT_HERSHEY_SIMPLEX
    with MappedArray(request, "main") as m:
        current_time = time.time()
        elapsed_time = current_time - start_time
        status_text = "Scanning barcodes..." if not ocr_mode else "OCR MODE ACTIVE"
        cv2.putText(m.array, status_text, (10, 30), font, 0.7, (0, 0, 255), 2)
        for b in barcodes:
            if b.polygon:
                points = np.array([(p.x, p.y) for p in b.polygon])
                cv2.polylines(m.array, [points], True, colour_barcode, 2)
                x, y = min(p.x for p in b.polygon), min(p.y for p in b.polygon) - 30
                cv2.putText(m.array, b.data.decode('utf-8'), (x, y), font, 0.7, colour_barcode, 2)
        if ocr_mode:
            for item in display_data:
                x, y, w, h = item["box"]
                cv2.putText(m.array, item["text"], (x, y), font, 0.7, colour_ocr, 2)

def main():
    global start_time, barcodes, display_data, ocr_mode
    print("Starting Raspberry Pi barcode scanner with OCR fallback...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 960)})
    picam2.configure(config)
    picam2.start_preview(Preview.QTGL)
    picam2.post_callback = draw_overlay
    picam2.start()
    start_time = time.time()
    barcodes = []
    display_data = []
    ocr_mode = False
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
            if not ocr_mode and elapsed_time > 30:
                print("No barcode detected for 30 seconds. Activating OCR mode.")
                ocr_mode = True
            barcode_detected = False
            barcodes = decode(frame)
            if barcodes:
                for barcode in barcodes:
                    barcode_data = barcode.data.decode('utf-8')
                    if barcode_data not in detected_barcodes or (current_time - detected_barcodes[barcode_data] > cooldown_time):
                        print(f"Time: {time.strftime('%H:%M:%S')}")
                        print(f"Barcode: {barcode_data}")
                        print("-------------------")
                        detected_barcodes[barcode_data] = current_time
                        with open("scanned_barcodes.txt", "a") as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {barcode_data}\n")
                        beep()
                        start_time = current_time
                        ocr_mode = False
                        barcode_detected = True
            if ocr_mode and not barcode_detected and (current_time - last_ocr_time > ocr_cooldown):
                print("Attempting OCR to extract ID number...")
                ocr_data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
                extracted_data = []
                for i in range(len(ocr_data['text'])):
                    if ocr_data['text'][i].strip():
                        conf = int(ocr_data['conf'][i]) if ocr_data['conf'][i].isdigit() else 0
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
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        picam2.stop_preview()
        picam2.stop()
        print("Barcode scanner closed")

start_time = 0
barcodes = []
display_data = []
ocr_mode = False

if __name__ == "__main__":
    main()
