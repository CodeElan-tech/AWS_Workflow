import cv2
import yt_dlp
import numpy as np
from flask import Flask, render_template, Response, request
from ultralytics import YOLO  # Import YOLO from Ultralytics
import subprocess

# Initialize Flask app
app = Flask(__name__)
video_capture = None

# Path to your cookies file (exported manually or using --cookies-from-browser)
cookies_file = "youtube_cookies.txt"

# yt-dlp command to fetch the direct stream URL
def get_stream_url(youtube_url):
    try:
        command = ['yt-dlp', '-g', '--cookies', cookies_file, youtube_url]
        # Run the command to fetch the stream URL
        stream_url = subprocess.check_output(command, stderr=subprocess.STDOUT).decode().strip()
        return stream_url
    except Exception as e:
        print(f"Error fetching stream URL: {e}")
        return None

# Load YOLOv8 model from Ultralytics (YOLOv8n is a lightweight version)
model = YOLO('yolov8n.pt')  # Load the YOLOv8n model from Ultralytics

# Function to detect objects in the frame using YOLOv8
def detect_objects(frame):
    # Perform object detection on the frame using YOLOv8 model
    results = model(frame)  # This will return the detection results
    # Annotate frame with bounding boxes and labels
    annotated_frame = results[0].plot()  # This renders the frame with bounding boxes
    return annotated_frame

# Video feed generator with YOLOv8 object detection
def generate_frames():
    global video_capture
    while video_capture:
        success, frame = video_capture.read()
        if not success:
            break

        # Apply YOLO object detection to the frame
        frame = detect_objects(frame)

        # Encode the frame as JPEG for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    global video_capture

    stream_started = False
    youtube_url = None
    error_message = None

    if request.method == 'POST':
        youtube_url = request.form['youtube_url']
        stream_url = get_stream_url(youtube_url)

        if stream_url:
            video_capture = cv2.VideoCapture(stream_url)
            stream_started = True
        else:
            error_message = "Failed to fetch the stream URL. Please try again."

    return render_template('index.html', stream_started=stream_started, youtube_url=youtube_url, error_message=error_message)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


# Route to stop video capture
@app.route("/stop/<camera_id>")
def stop_video(camera_id):
    global video_capture
    with lock:
        if camera_id in video_capture:
            cap, _ = video_capture.pop(camera_id)
            cap.release()

    return f"Camera {camera_id} stopped."


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)


