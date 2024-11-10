import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import numpy as np
import argparse
import mlflow
from ultralytics import YOLO
from deep_sort.deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.deep_sort.detection import Detection as DeepSortDetection

# Define a mapping for object names to colors
OBJECT_COLORS = {
    'car': (0, 255, 0),        # Green for cars
    'motorcycle': (255, 0, 0), # Blue for motorcycles
    'truck': (0, 255, 255),    # Yellow for trucks
    'bus': (255, 255, 0),      # Light Blue for buses
    'bicycle': (128, 128, 0),  # Olive for bicycles
    'person': (255, 255, 255)  # White for people
}

vehicle_count = {key: 0 for key in OBJECT_COLORS.keys()}
counted_ids = set()

def adjust_for_day_or_night(frame, time_of_day):
    if time_of_day == "night":
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
    return frame

def detect_vehicles(model_path, video_path, output_video_path, selected_types, time_of_day):
    # Start an MLflow run
    with mlflow.start_run(run_name="mlflow_vehicle_detection") as run:
        # Log input parameters to MLflow
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("video_path", video_path)
        mlflow.log_param("output_video_path", output_video_path)
        mlflow.log_param("selected_types", selected_types)
        mlflow.log_param("time_of_day", time_of_day)

        # Log initial metrics (vehicle count as 0)
        for key in vehicle_count:
            mlflow.log_metric(f"{key}_count", 0)

        # Load the YOLOv8 model
        model_yolov8 = YOLO(model_path)

        # Initialize the video capture
        cap = cv2.VideoCapture(video_path)

        # Check if video is opened successfully
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return

        # Output video settings
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Initialize the DeepSORT tracker
        metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.4, budget=100)
        tracker = Tracker(metric, max_age=15, n_init=3)

        track_labels = {}

        frame_count = 0  # Track number of frames processed
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_count}")
                break

            frame = adjust_for_day_or_night(frame, time_of_day)
            frame_count += 1

            # Perform object detection using YOLOv8
            results_yolov8 = model_yolov8(frame)

            detections = []
            detection_boxes = []

            # Process YOLOv8 results
            for result in results_yolov8:
                for detection in result.boxes:
                    x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
                    conf = detection.conf[0].item()
                    cls = detection.cls[0].item()
                    label = model_yolov8.names[int(cls)]

                    # Check if the detected object is among the selected vehicle types
                    if label in selected_types and conf > 0.6:
                        width = x2 - x1
                        height = y2 - y1

                        detection_obj = DeepSortDetection(tlwh=np.array([x1, y1, width, height]), confidence=conf,
                                                          feature=np.random.rand(128).astype(np.float32))
                        detections.append(detection_obj)
                        detection_boxes.append((x1, y1, x2, y2, label))

                        track_labels[len(detections) - 1] = label

            # Update tracker with the current detections
            tracker.predict()
            tracker.update(detections)

            # Iterate through tracks and draw bounding boxes
            for track in tracker.tracks:
                if not track.is_confirmed():
                    continue

                bbox = track.to_tlbr()
                track_id = track.track_id

                if track_id not in track_labels:
                    for (x1, y1, x2, y2, label) in detection_boxes:
                        if abs(bbox[0] - x1) < 20 and abs(bbox[1] - y1) < 20 and abs(bbox[2] - x2) < 20 and abs(bbox[3] - y2) < 20:
                            track_labels[track_id] = label
                            break

                label = track_labels.get(track_id, 'Unknown')

                # Count detected vehicles
                if label in vehicle_count and track_id not in counted_ids:
                    vehicle_count[label] += 1
                    counted_ids.add(track_id)

                # Draw bounding box and label
                color = OBJECT_COLORS.get(label, (0, 255, 0))
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, f'{label} ID: {track_id}', (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Display counts on the frame
            count_text = ', '.join([f'{k}: {v}' for k, v in vehicle_count.items() if v > 0])
            cv2.putText(frame, f'Counts: {count_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Write frame to output video
            out.write(frame)

            # Log metrics every 10 frames to avoid overwhelming the UI
            if frame_count % 10 == 0:
                for key, count in vehicle_count.items():
                    mlflow.log_metric(f"{key}_count", count, step=frame_count)

        # Log final vehicle count as metrics
        for key, count in vehicle_count.items():
            mlflow.log_metric(f"{key}_count_final", count)

        # Log additional metrics
        mlflow.log_metric("total_frames_processed", frame_count)

        # Log the output video as an artifact
        out.release()
        mlflow.log_artifact(output_video_path)

        # Capture a screenshot of the last processed frame as an artifact
        screenshot_path = "last_frame_screenshot.jpg"
        if frame is not None and frame.size > 0:
            cv2.imwrite(screenshot_path, frame)
            mlflow.log_artifact(screenshot_path)
        else:
            print(f"No valid frame to capture at frame count {frame_count}, skipping screenshot.")

        cap.release()
        print(f"Output video saved to: {output_video_path}")
        print("Vehicle detection completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Detection and Tracking")
    parser.add_argument("model_path", type=str, help="Path to YOLO model")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("output_video_path", type=str, help="Path to save the output video")
    parser.add_argument("time_of_day", type=str, choices=["day", "night"], help="Time of day: 'day' or 'night'")
    parser.add_argument("--selected_types", type=str, nargs="+", default=['car', 'motorcycle', 'truck', 'bus', 'bicycle', 'person'], help="Types of objects to detect")
    args = parser.parse_args()

    detect_vehicles(args.model_path, args.video_path, args.output_video_path, args.selected_types, args.time_of_day)
