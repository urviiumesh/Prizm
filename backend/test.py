import cv2
import numpy as np
import os
import time
from flask import Flask, Response, request, jsonify
import easyocr
import google.generativeai as genai  # Import Gemini API library
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app, origins="http://127.0.0.1:5500")
# Automatically create the folder to store images
CAPTURE_FOLDER = "captured_images"
if not os.path.exists(CAPTURE_FOLDER):
    os.makedirs(CAPTURE_FOLDER)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# State for controlling scanning
scanning_state = {"active": True, "captured_texts": []}

# Function to detect edges and approximate document contours
def detect_document_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological closing
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(closing, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10000:  # Filter small contours
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 2:  # Filter aspect ratio
                return approx
    return None



def guide_user(approx, frame_width, frame_height, capturing_state):  # Add capturing_state to function signature
    cx, cy = np.mean(approx[:, 0, :], axis=0)
    if cx < frame_width * 0.3:  
        return "Move paper to the right.", False
    elif cx > frame_width * 0.7:
        return "Move paper to the left.", False
    elif cy < frame_height * 0.3:
        return "Move paper down.", False
    elif cy > frame_height * 0.7:
        return "Move paper up.", False
    else:
        capturing_state["captured"] = True
        return "Paper aligned. Capturing...", True


def process_frame(frame, last_capture_time, capturing_state, capture_delay=1):
    frame_height, frame_width = frame.shape[:2]

    if capturing_state["captured"]:
        elapsed_time = time.time() - capturing_state["capture_time"]
        if elapsed_time < 6:  # Show "Captured" message for 6 seconds
            cv2.putText(frame, "Document captured!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            extracted_text = " ".join(scanning_state["captured_texts"])
            cv2.putText(frame, extracted_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame, last_capture_time
        else:
            capturing_state["captured"] = False

    approx = detect_document_contour(frame)

    if approx is not None:
        guide_message, aligned = guide_user(approx, frame_width, frame_height, capturing_state)  # Corrected
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

        # Extract the bounding rectangle for the document
        rect = cv2.boundingRect(approx)
        x, y, w, h = rect  # Ensure that these values are initialized correctly

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.putText(frame, guide_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        current_time = time.time()
        if aligned and current_time - last_capture_time > capture_delay:
            # Crop the detected document
            cropped_frame = frame[y:y+h, x:x+w]
            filename = f"{CAPTURE_FOLDER}/captured_image_{int(current_time)}.jpg"
            cv2.imwrite(filename, cropped_frame)
            print(f"Image captured and saved as {filename}")

            # Extract text using EasyOCR
            extracted_text = reader.readtext(cropped_frame, detail=0)
            scanning_state["captured_texts"].append(" ".join(extracted_text))
            print(scanning_state["captured_texts"])
            capturing_state["captured"] = True
            capturing_state["capture_time"] = current_time
            last_capture_time = current_time
    else:
        cv2.putText(frame, "No document detected. Adjust paper.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, last_capture_time




@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message', '')  # Get message from frontend

    if not user_input:
        return jsonify({"error": "No message provided."}), 400

    # Add the user input to captured texts
    full_text = " ".join(scanning_state["captured_texts"]) + " " + user_input

    # Send to Gemini API and get the response
    summary = summarize_with_gemini_api(full_text)

    return jsonify({"response": summary})

# Summarization function using Gemini API
def summarize_with_gemini_api(text):
    # Configure API key
    genai.configure(api_key="AIzaSyATkIjfp2GQy6lc45sYbN66UoZBMTrPBXY")  # Replace with your valid API key

    model = genai.GenerativeModel("gemini-1.5-flash")
    
    response = model.generate_content("Summarize this if you think it has a summary and point out whether this is exploitative in nature or not, if there is nothing to summarize, answer the question briefly, do not do both, begin the response with it is a reply if you are answering it: here is the" + text)
    
    print(response.text)
    # Return the generated text
    return response.text if 'reply' in response.text else "No summary available."

# New endpoint to summarize the scanned texts
@app.route('/summarize_text', methods=['GET'])
def summarize_text():
    # Combine all captured texts into a single string
    full_text = " ".join(scanning_state["captured_texts"])

    if not full_text:
        return jsonify({"error": "No text available for summarization."}), 400

    # Summarize using Gemini API
    summary = summarize_with_gemini_api(full_text)
    return jsonify({"summary": summary})


@app.route('/camera')
def camera_feed():
    def generate():
        cap = cv2.VideoCapture(0)  # Open camera
        last_capture_time = time.time()  # Initialize last capture time
        
        # Initialize capturing_state
        capturing_state = {
            "captured": False,
            "capture_time": time.time()
        }

        while True:
            ret, frame = cap.read()  # Read frame from camera

            if not ret:  # Check for failed capture
                print("Failed to capture frame from camera.")
                break

            # Now pass capturing_state to the process_frame function
            frame, last_capture_time = process_frame(frame, last_capture_time, capturing_state)  # Pass capturing_state here

            # Encode frame as JPEG and convert to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        cap.release()  # Release camera resources

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file
        file_path = os.path.join(CAPTURE_FOLDER, file.filename)
        file.save(file_path)

        # Process the image with EasyOCR
        image = cv2.imread(file_path)
        extracted_text = reader.readtext(image, detail=0)

        # Add the extracted text to captured texts
        scanning_state["captured_texts"].append(" ".join(extracted_text))

        # Return the extracted text and summary
        full_text = " ".join(scanning_state["captured_texts"])
        summary = summarize_with_gemini_api(full_text)

        return jsonify({
            "extracted_text": " ".join(scanning_state["captured_texts"]),
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/scanned_texts', methods=['GET'])
def get_scanned_texts():
    return jsonify({"texts": scanning_state["captured_texts"]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
