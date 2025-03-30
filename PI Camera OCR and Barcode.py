import cv2
import time
import numpy as np
import os
import pytesseract
import re
from pyzbar.pyzbar import decode
from picamera2 import MappedArray, Picamera2, Preview

# For beep sound on Raspberry Pi
def beep():
    print('\a')  # Print alert character
    # Alternative for Raspberry Pi (uncomment if needed)
    # import os
    # os.system('echo -e "\a"')

# Function to extract ID number using OCR
def extract_id_number(frame):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to improve OCR
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(thresh)
        
        # Look for ID Number pattern (format like "ID Number: 9999999")
        id_match = re.search(r'ID\s*Number:?\s*(\d+)', text, re.IGNORECASE)
        if id_match:
            return id_match.group(1)  # Return just the number
        
        # If no match with "ID Number:", try to find any sequence of digits that could be an ID
        digits_match = re.search(r'(\d{7,})', text)  # Looking for 7+ digit numbers
        if digits_match:
            return digits_match.group(1)
            
        return None
    except Exception as e:
        print(f"OCR error: {e}")
        return None

# Function to draw barcodes on the image
def draw_overlay(request):
    global display_data, barcodes, ocr_mode
    
    colour_barcode = (0, 255, 0)  # Green
    colour_ocr = (0, 255, 255)    # Yellow
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    with MappedArray(request, "main") as m:
        # Add status text
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        status_text = "Scanning barcodes..."
        if ocr_mode:
            status_text = "OCR MODE ACTIVE"
        else:
            status_text += f" (OCR in {max(0, 30 - int(elapsed_time))}s)"
        
        cv2.putText(m.array, status_text, (10, 30), font, 0.7, (0, 0, 255), 2)
        
        # Draw barcode information
        for b in barcodes:
            if b.polygon:
                # Get polygon points
                points = np.array([(p.x, p.y) for p in b.polygon])
                
                # Draw polygon around the barcode
                cv2.polylines(m.array, [points], True, colour_barcode, 2)
                
                # Draw data
                x = min([p.x for p in b.polygon])
                y = min([p.y for p in b.polygon]) - 30
                cv2.putText(m.array, b.data.decode('utf-8'), (x, y), font, 0.7, colour_barcode, 2)
        
        # Draw OCR information
        if ocr_mode:
            for item in display_data:
                x, y, w, h = item["box"]
                if item.get("is_id", False):
                    # Highlight ID numbers differently
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(m.array, f"ID: {item['text']}", (x, y - 10), font, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(m.array, item["text"], (x, y), font, 0.7, colour_ocr, 2)

def main():
    global start_time, barcodes, display_data, ocr_mode
    
    print("Starting Raspberry Pi barcode scanner with OCR fallback...")
    
    # Initialize camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 960)})
    picam2.configure(config)
    picam2.start_preview(Preview.QTGL)
    
    # Set post-processing callback
    picam2.post_callback = draw_overlay
    
    # Start the camera
    picam2.start()
    
    # Initialize variables
    start_time = time.time()
    barcodes = []
    display_data = []
    ocr_mode = False
    
    # Dictionary to track last scanned barcodes and timestamps
    detected_barcodes = {}
    cooldown_time = 2  # seconds
    
    # OCR parameters
    ocr_threshold = 50  # Confidence threshold for OCR
    ocr_cooldown = 5  # seconds between OCR attempts
    last_ocr_time = 0
    
    print("Scanner started. Press Ctrl+C to quit.")
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array("main")
            
            # Track time
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check if we need to activate OCR mode after 30 seconds with no barcode
            if not ocr_mode and elapsed_time > 30:
                print("No barcode detected for 30 seconds. Activating OCR mode.")
                ocr_mode = True
            
            barcode_detected = False
            
            # Process frame with pyzbar for barcode detection
            barcodes = decode(frame)
            
            # Process any detected barcodes
            if barcodes:
                for barcode in barcodes:
                    # Extract barcode data and format
                    barcode_data = barcode.data.decode('utf-8')
                    barcode_format = barcode.type
                    
                    # Only process if this is a new barcode or enough time has passed
                    if barcode_data not in detected_barcodes or (current_time - detected_barcodes[barcode_data] > cooldown_time):
                        print(f"Time: {time.strftime('%H:%M:%S')}")
                        print(f"Barcode: {barcode_data}")
                        print(f"Type: {barcode_format}")
                        print("-------------------")
                        
                        detected_barcodes[barcode_data] = current_time
                        
                        # Save barcode data to a file
                        with open("scanned_barcodes.txt", "a") as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {barcode_data} ({barcode_format})\n")
                        
                        # Beep sound on detection
                        beep()
                        
                        # Reset timer since we found a barcode
                        start_time = current_time
                        ocr_mode = False
                        barcode_detected = True
            
            # If OCR mode is active and no barcode was detected, try OCR (with cooldown)
            if ocr_mode and not barcode_detected and (current_time - last_ocr_time > ocr_cooldown):
                print("Attempting OCR to extract ID number...")
                
                # Get OCR data with confidence scores
                data = [line.split('\t') for line in pytesseract.image_to_data(frame).split('\n')][1:-1]
                data = [{"text": item[11], "conf": int(item[10]), "box": (item[6], item[7], item[8], item[9])} 
                       for item in data if len(item) >= 12]  # Ensure we have enough columns
                
                # Filter OCR results by confidence threshold
                data = [item for item in data if item["conf"] > ocr_threshold and not item["text"].isspace()]
                
                # Convert box coordinates to integers
                for item in data:
                    item["box"] = tuple(map(int, item["box"]))
                
                # Look for ID numbers in the OCR results
                id_number = extract_id_number(frame)
                if id_number:
                    print(f"Time: {time.strftime('%H:%M:%S')}")
                    print(f"ID Number (via OCR): {id_number}")
                    print("-------------------")
                    
                    # Save ID data to the same file
                    with open("scanned_barcodes.txt", "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ID Number: {id_number} (OCR)\n")
                    
                    # Beep sound on detection
                    beep()
                    
                    # Mark the ID in the display data
                    for item in data:
                        if id_number in item["text"]:
                            item["is_id"] = True
                
                # Update display data for all OCR text
                display_data = data
                
                # Update OCR attempt time
                last_ocr_time = current_time
                    
            # Short delay to reduce CPU usage
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Close the camera and clean up
        picam2.stop_preview()
        picam2.stop()
        print("Barcode scanner closed")

# Global variables (needed for callback)
start_time = 0
barcodes = []
display_data = []
ocr_mode = False

if __name__ == "__main__":
    main()