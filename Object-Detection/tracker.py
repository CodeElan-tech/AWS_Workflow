import cv2
import numpy as np
import mlflow
# import mlflow.log_metrics
from ultralytics import YOLO
from deep_sort.deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.deep_sort.detection import Detection as DeepSortDetection
import os

# Define a mapping for object names to colors
OBJECT_COLORS = {
    'car': (0, 255, 0),       # Green for cars
    'motorcycle': (255, 0, 0), # Blue for motorcycles
    'truck': (0, 255, 255),   # Yellow for trucks
    'bus': (255, 255, 0),     # Light Blue for buses
    'bicycle': (128, 128, 0), # Olive for bicycles
    'person': (255, 255, 255) # White for people
}

def detect_vehicles(video_path):
    # Start the MLflow run
    with mlflow.start_run(run_name="mlflow_vehicle_detection") as run:
        # Load the YOLO model
        model = YOLO('yolov8l.pt')  # Load the YOLOv8 model

        # Initialize the video capture
        cap = cv2.VideoCapture(video_path)

        # Output video settings
        output_video_path = 'output_tracked3.avi'  # Define output video path
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for output video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Initialize the DeepSORT tracker
        metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.4, budget=100)
        tracker = Tracker(metric)

        # Initialize a dictionary to count detected objects
        vehicle_count = {key: 0 for key in OBJECT_COLORS.keys()}
        counted_ids = set()

        track_labels = {}

        # Process each frame of the video
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection
            results = model(frame)

            # Prepare detections for DeepSORT
            detections = []
            detection_boxes = []

            for result in results:
                for detection in result.boxes:
                    x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int) # Bounding box coordinates
                    conf = detection.conf[0].item()  # Confidence score
                    cls = detection.cls[0].item() # Class index
                    label = model.names[int(cls)]  # Get the label

                    # Debug prints
                    print(f'Detected: {label} (Confidence: {conf})')

                    # Check if the detected object is among the specified vehicle types
                    if label in OBJECT_COLORS and conf > 0.6:  # Adjust the confidence threshold as needed
                        width = x2 - x1
                        height = y2 - y1
                        detection_obj = DeepSortDetection(tlwh=np.array([x1, y1, width, height]), confidence=conf, feature= np.random.rand(128).astype(np.float32))
                        detections.append(detection_obj)
                        detection_boxes.append((x1, y1, x2, y2, label))

                        track_labels[len(detections) - 1] = label 

            # Update tracker with the current detections
            tracker.predict()
            tracker.update(detections)

            # Iterate through tracks and draw bounding boxes
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()  # Convert track to (top left, bottom right) format
                track_id = track.track_id

                if track_id not in track_labels:
                    for (x1, y1, x2, y2, label) in detection_boxes:
                        # Compare the detection bounding box to the track bounding box to find a match
                        if abs(bbox[0] - x1) < 20 and abs(bbox[1] - y1) < 20 and abs(bbox[2] - x2) < 20 and abs(bbox[3] - y2) < 20:
                            track_labels[track_id] = label  # Assign the label to the track ID
                            break

                # Get the label for the current track
                label = track_labels.get(track_id, 'Unknown')

                if label in vehicle_count and track_id not in counted_ids:
                    vehicle_count[label] += 1  # Count detected vehicles
                    counted_ids.add(track_id)  # Count detected vehicles

                # Draw bounding box and label
                color = OBJECT_COLORS.get(label, (0, 255, 0))  # Default color if not found
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, f'{label} ID: {track_id}', (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Display the count of each vehicle type at the top-left corner
            y_offset = 30
            for vehicle, count in vehicle_count.items():
                cv2.putText(frame, f'{vehicle}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 20

            # Write the frame with detections to the output video
            out.write(frame)

            # Log metrics for each frame
            frame_num += 1
            for vehicle, count in vehicle_count.items():
                mlflow.log_metric(f'{vehicle}_count', count, step=frame_num)

        # Release video resources
        cap.release()
        out.release()

        # Log the output video as an artifact
        mlflow.log_artifact(output_video_path)

        # Log the final vehicle count
        mlflow.log_metrics(vehicle_count)

        # Log the run details
        print(f"Output video saved to: {output_video_path}")
        print("MLflow run completed.")
