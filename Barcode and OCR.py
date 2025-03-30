import cv2
import zxingcpp
import time
import numpy as np
import os
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as needed

# For beep sound - platform independent
def beep():
    if os.name == 'nt':  # Windows
        import winsound
        winsound.Beep(1000, 200)  # 1000 Hz for 200ms
    else:  # Linux/Mac/Raspberry Pi
        print('\a')  # Print alert character

# Disable MSMF backend to prevent webcam issues (for Windows)
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE"] = "0"

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

def scan_barcodes():
    print("Starting high-resolution barcode scanner with OCR fallback...")
    
    # Initialize webcam - platform specific
    if os.name == 'nt':  # Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:  # Linux/Mac/Raspberry Pi
        cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully")
    
    # Try to set the highest resolution possible
    # Start with HD resolution
    resolutions = [
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
        (1024, 768),   # XGA
        (800, 600),    # SVGA
        (640, 480)     # VGA - fallback
    ]
    
    success = False
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Check if resolution was set successfully
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if abs(actual_width - width) < 100 and abs(actual_height - height) < 100:
            print(f"Successfully set resolution to {int(actual_width)}x{int(actual_height)}")
            success = True
            break
    
    if not success:
        print("Failed to set specific resolution. Using webcam default.")
    
    print(f"Current resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("Press 'q' to quit")
    
    # Set focus to auto if available
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    # Try to increase sharpness if available
    cap.set(cv2.CAP_PROP_SHARPNESS, 100)
    
    # Create an OpenCV window
    cv2.namedWindow("Barcode Scanner", cv2.WINDOW_NORMAL)
    
    # Dictionary to track last scanned barcodes and timestamps
    detected_barcodes = {}
    cooldown_time = 2  # seconds
    
    # Add variables for OCR fallback after 30 seconds
    start_time = time.time()
    ocr_timer_active = True
    ocr_cooldown = 5  # seconds between OCR attempts
    last_ocr_time = 0
    ocr_mode = False
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # Check if frame was successfully captured
            if not ret or frame is None:
                print("Error: Can't receive frame. Exiting...")
                break
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Calculate elapsed time
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check if we need to activate OCR mode after 30 seconds with no barcode
            if ocr_timer_active and elapsed_time > 30 and not ocr_mode:
                print("No barcode detected for 30 seconds. Activating OCR mode.")
                ocr_mode = True
            
            barcode_detected = False
            
            # Try regular barcode detection first
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply slight Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding for better contrast
            enhanced = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Try detection on both original and enhanced images
            results = zxingcpp.read_barcodes(frame)
            if not results:
                results = zxingcpp.read_barcodes(enhanced)
            
            # Process any detected barcodes
            if results:
                for result in results:
                    # Extract barcode data and format
                    barcode_data = result.text
                    barcode_format = result.format.name
                    
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
                    
                    # Get the position points of the barcode and highlight it
                    try:
                        position = result.position
                        if position:
                            # Try to convert position directly to points
                            points = []
                            for point in position:
                                points.append([int(point.x), int(point.y)])
                            
                            # Convert to numpy array
                            if points:
                                pts = np.array(points, np.int32)
                                pts = pts.reshape((-1, 1, 2))
                                
                                # Draw polygon around the barcode
                                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                                
                                # Get the top-left corner for text placement
                                # Use the first point for simplicity
                                x, y = points[0]
                                y = max(y - 10, 10)  # Ensure text isn't off screen
                                
                                # Put barcode data and type on the frame
                                text = f"{barcode_data} ({barcode_format})"
                                cv2.putText(display_frame, text, (x, y), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Error highlighting barcode: {e}")
                        # Continue even if we can't highlight the barcode
            
            # If OCR mode is active and no barcode was detected, try OCR (with cooldown)
            if ocr_mode and not barcode_detected and (current_time - last_ocr_time > ocr_cooldown):
                print("Attempting OCR to extract ID number...")
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
                    
                    # Display ID on frame
                    cv2.putText(display_frame, f"ID: {id_number} (OCR)", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update OCR attempt time
                last_ocr_time = current_time
            
            # Display enhanced image in corner for debugging
            h, w = display_frame.shape[:2]
            small_enhanced = cv2.resize(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), (w//4, h//4))
            display_frame[h-h//4:h, 0:w//4] = small_enhanced
            
            # Add scanning status text
            status_text = f"Scanning... ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})"
            if ocr_mode:
                status_text += " | OCR MODE ACTIVE"
            else:
                status_text += f" | OCR in {max(0, 30 - int(elapsed_time))}s"
            status_text += " | Press 'q' to quit"
            
            cv2.putText(display_frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow("Barcode Scanner", display_frame)
            
            # Break the loop if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        print("Barcode scanner closed")

if __name__ == "__main__":
    scan_barcodes()