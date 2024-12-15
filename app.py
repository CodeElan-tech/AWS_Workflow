import os
import subprocess
import cv2
from flask import Flask, render_template, Response, request, session
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using session

# Initialize the YOLO model (ensure you've loaded the correct model)
model = YOLO('yolov8n.pt')  # Change this to your model path

def get_youtube_stream_url(youtube_url):
    if youtube_url is None or youtube_url == '':
        raise ValueError("YouTube URL must not be None or empty.")

    # Construct your command (using yt-dlp)
    command = ['yt-dlp', '-f', 'best', '-g', youtube_url]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stream_url = result.stdout.strip()
    return stream_url

def generate_frames(youtube_url):
    """
    Generate frames from a YouTube live stream URL and perform YOLO detection.
    """
    # Fetch the direct stream URL
    stream_url = get_youtube_stream_url(youtube_url)
    if not stream_url:
        print("Error: Unable to retrieve the YouTube stream URL.")
        return

    # Open the live stream
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Unable to open the live stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the stream.")
            break

        # Perform YOLO detection
        results = model.predict(frame, conf=0.5, stream=True)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Render the main HTML page and process the form.
    """
    youtube_links = []  # Default empty list of YouTube links
    if request.method == 'POST':
        # Get the list of YouTube links from the form
        try:
            link_count = int(request.form['link-count'])  # Get the number of links submitted
            youtube_links = [request.form[f'link-{i}'] for i in range(link_count)]  # Collect the links

            # Store the list of YouTube links in session
            session['youtube_links'] = youtube_links

        except KeyError:
            return "Error: Invalid form data", 400  # Handle missing form data gracefully

    # Render the page with previously entered YouTube links (if any)
    return render_template('index.html', youtube_links=youtube_links)

@app.route('/video_feed/<int:index>')
def video_feed(index):
    """
    Route for streaming the video feed for the given index (i.e., the link).
    """
    youtube_links = session.get('youtube_links', [])
    if index < 0 or index >= len(youtube_links):
        return "Error: Invalid video index", 404

    youtube_url = youtube_links[index]  # Get the appropriate YouTube URL from session
    return Response(generate_frames(youtube_url), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

