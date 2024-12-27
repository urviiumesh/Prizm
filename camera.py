import cv2
import easyocr
import os
import time

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Folder to save captured images
CAPTURE_FOLDER = "capture_images"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# State dictionaries
capturing_state = {"captured": False, "capture_time": 0}
scanning_state = {"captured_texts": []}

# Configuration
capture_delay = 2  # Seconds between captures
last_capture_time = time.time()

# Define guidance logic
def guide_user(cx, cy, frame_width, frame_height):
    if cx < frame_width * 0.3:
        return "Move paper to the right.", False
    elif cx > frame_width * 0.7:
        return "Move paper to the left.", False
    elif cy < frame_height * 0.3:
        return "Move paper down.", False
    elif cy > frame_height * 0.7:
        return "Move paper up.", False
    else:
        return "Paper aligned. Capturing...", True

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from camera.")
            break

        frame_height, frame_width, _ = frame.shape

        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        aligned = False
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:  # Document detected
                x, y, w, h = cv2.boundingRect(approx)
                cx, cy = x + w // 2, y + h // 2

                # Draw bounding box
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

                # Guide user and check alignment
                message, aligned = guide_user(cx, cy, frame_width, frame_height)
                cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if aligned:
                    current_time = time.time()
                    if current_time - last_capture_time > capture_delay:
                        # Crop and save the detected document
                        cropped_frame = frame[y:y+h, x:x+w]
                        filename = f"{CAPTURE_FOLDER}/captured_image_{int(current_time)}.jpg"
                        cv2.imwrite(filename, cropped_frame)
                        print(f"Image captured and saved as {filename}")

                        # Extract text using EasyOCR
                        extracted_text = reader.readtext(cropped_frame, detail=0)
                        scanning_state["captured_texts"].append(" ".join(extracted_text))
                        print("Extracted Text:", scanning_state["captured_texts"][-1])

                        # Update state
                        capturing_state["captured"] = True
                        capturing_state["capture_time"] = current_time
                        last_capture_time = current_time

                break  # Only process the largest detected contour

        # Display the camera feed
        cv2.imshow("Document Scanner", frame)

        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on 'q'
            break
        elif key == ord('d'):  # User input 'done'
            user_input = input("Type 'done' to stop capturing or press Enter to continue: ").strip().lower()
            if user_input == "done":
                print("Stopping capture as per user input.")
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Captured texts:")
for text in scanning_state["captured_texts"]:
    print(text)
