import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort.deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.deep_sort.detection import Detection as DeepSortDetection
import boto3

OBJECT_COLORS = {
    'car': (0, 255, 0),        # Green for cars
    'motorcycle': (255, 0, 0), # Blue for motorcycles
    'truck': (0, 255, 255),    # Yellow for trucks
    'bus': (255, 255, 0),      # Light Blue for buses
    'bicycle': (128, 128, 0),  # Olive for bicycles
    'person': (255, 255, 255), # White for people
    'autorickshaw': (255, 0, 255) # Purple for auto-rickshaws
}

vehicle_count = {key: 0 for key in OBJECT_COLORS.keys()}
counted_ids = set()

def adjust_for_day_or_night(frame, time_of_day):
    if time_of_day == "night":
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
    return frame

def detect_vehicles(video_path, selected_types, time_of_day):
    model_yolov8 = YOLO('yolov8l.pt')
    model_autorickshaw = YOLO('IndianVehicledetectionmodel.pt')

    cap = cv2.VideoCapture(video_path)

    input_video_name = os.path.splitext(os.path.basename(video_path))[0]
    local_output_video = f'{input_video_name}_output.avi'

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(local_output_video, fourcc, fps, (frame_width, frame_height))

    metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.4, budget=100)
    tracker = Tracker(metric, max_age=15, n_init=3)

    track_labels = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = adjust_for_day_or_night(frame, time_of_day)
        results_yolov8 = model_yolov8(frame)
        results_autorickshaw = model_autorickshaw(frame)

        detections = []
        detection_boxes = []

        for result in results_yolov8:
            for detection in result.boxes:
                x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
                conf = detection.conf[0].item()
                cls = detection.cls[0].item()
                label = model_yolov8.names[int(cls)]

                if label in selected_types and conf > 0.6:
                    width = x2 - x1
                    height = y2 - y1
                    detection_obj = DeepSortDetection(tlwh=np.array([x1, y1, width, height]), confidence=conf,
                                                      feature=np.random.rand(128).astype(np.float32))
                    detections.append(detection_obj)
                    detection_boxes.append((x1, y1, x2, y2, label))
                    track_labels[len(detections) - 1] = label

        for result in results_autorickshaw:
            for detection in result.boxes:
                x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
                conf = detection.conf[0].item()
                label = 'autorickshaw'

                if conf > 0.6:
                    width = x2 - x1
                    height = y2 - y1
                    detection_obj = DeepSortDetection(tlwh=np.array([x1, y1, width, height]), confidence=conf,
                                                      feature=np.random.rand(128).astype(np.float32))
                    detections.append(detection_obj)
                    detection_boxes.append((x1, y1, x2, y2, label))
                    track_labels[len(detections) - 1] = label

        tracker.predict()
        tracker.update(detections)

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

            if label in vehicle_count and track_id not in counted_ids:
                vehicle_count[label] += 1
                counted_ids.add(track_id)

            color = OBJECT_COLORS.get(label, (0, 255, 0))
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, f'{label} ID: {track_id}', (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        count_text = ', '.join([f'{k}: {v}' for k, v in vehicle_count.items() if v > 0])
        cv2.putText(frame, f'Counts: {count_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    s3 = boto3.client(
        's3',
        aws_access_key_id='AKIA5FTZE3QEPV2D5VLK',
        aws_secret_access_key='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',
        region_name='eu-north-1'
    )
    bucket_name = 'shlokvideo'
    s3_key = f'{input_video_name}_output.avi'
    s3.upload_file(local_output_video, bucket_name, s3_key)
    print(f"Output video uploaded to S3 bucket: {bucket_name}/{s3_key}")

    os.remove(local_output_video)
    print("Local output video file deleted.")

video_path = r'D:\CodeElan-OpenCV\Videos\autoricksahw.mp4'
selected_types = ['car', 'motorcycle', 'truck', 'bus', 'bicycle', 'person', 'autorickshaw']
time_of_day = 'day'
detect_vehicles(video_path, selected_types, time_of_day)
