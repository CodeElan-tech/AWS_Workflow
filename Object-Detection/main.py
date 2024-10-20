from tracker import detect_vehicles  # Import the detect_vehicles function
import mlflow
from utils.utils import setup_mlflow, log_metrics, log_artifact, log_params, start_run, end_run

# Hardcoded video path
DEFAULT_VIDEO_PATH = 'H:/codeelan_git/Object-Detection-master/Object-Detection-master/sort_video.mp4'

def start_detection(video_path=DEFAULT_VIDEO_PATH):
    if not video_path:
        raise FileNotFoundError("Video file path is not provided.")

    # Start MLflow logging
    # setup_mlflow('vehicle-detection-experiment')
    # start_run()
    
    # Call the detect_vehicles function here
    detect_vehicles(video_path)

    # Optionally log artifacts or metrics
    # log_metrics(vehicle_count)
    # log_artifact(output_video_path)

    # End MLflow run
    # end_run()

def main():
    # Start vehicle detection with the default path
    try:
        start_detection()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
