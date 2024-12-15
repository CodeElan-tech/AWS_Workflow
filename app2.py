import cv2
import threading
import time
from flask import Flask, render_template, Response, request
import subprocess

# Initialize the Flask application
app = Flask(__name__)

# Global variables for video capture and threading
video_capture = {}
lock = threading.Lock()

# Function to get a YouTube stream URL using yt-dlp
def get_youtube_stream_url(youtube_url):
    """
    Fetch the direct stream URL for a YouTube video using yt-dlp with cookies.
    """
    if not youtube_url:
        raise ValueError("YouTube URL must not be empty.")

    # Path to your cookies file (exported manually or using --cookies-from-browser)
    cookies_file = "youtube_cookies.txt"

    # yt-dlp command to fetch the direct stream URL
    command = ['yt-dlp', '-g', '--cookies', cookies_file, youtube_url]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        stream_url = result.stdout.strip()

        # Ensure the result is valid
        if not stream_url:
            raise RuntimeError("yt-dlp returned an empty stream URL.")
        
        return stream_url

    except subprocess.CalledProcessError as e:
        error_message = f"Error fetching stream URL: {e.stderr.strip()}"
        print(error_message)
        raise RuntimeError(error_message)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise


# Function to capture video frames
def capture_video(camera_id, stream_url):
    global video_capture
    with lock:
        if camera_id in video_capture:
            return

    cap = cv2.VideoCapture(stream_url)
    video_capture[camera_id] = cap

    while True:
        with lock:
            if camera_id not in video_capture:
                break

        ret, frame = cap.read()
        if not ret:
            print(f"Camera {camera_id}: Unable to fetch frame. Retrying...")
            time.sleep(2)
            continue

        # Add your frame processing logic here if needed
        cv2.putText(frame, f"Camera {camera_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Store the frame in the global dictionary for video streaming
        with lock:
            video_capture[camera_id] = (cap, frame)


# Route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        youtube_url = request.form.get("youtube_url")
        camera_id = request.form.get("camera_id", "0")

        if not youtube_url:
            return "Error: YouTube URL is required."

        try:
            # Fetch the stream URL and start capturing video
            stream_url = get_youtube_stream_url(youtube_url)
            threading.Thread(target=capture_video, args=(camera_id, stream_url), daemon=True).start()
            return f"Camera {camera_id} started with YouTube URL: {youtube_url}"

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("index.html")


# Route to stream video feed
@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    def generate(camera_id):
        global video_capture
        while True:
            with lock:
                if camera_id not in video_capture:
                    break

                cap, frame = video_capture[camera_id]
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_data = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    return Response(generate(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")


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
    app.run(host='0.0.0.0', port=5000, debug=True)
