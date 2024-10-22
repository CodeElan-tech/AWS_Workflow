import tkinter as tk
from tkinter import filedialog, messagebox
from tracker import detect_vehicles  # Import the detect_vehicles function

class VehicleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Detection App")
        self.selected_vehicles = {}
        self.setup_gui()

    def setup_gui(self):
        # Video Selection
        video_label = tk.Label(self.root, text="Select Video File:")
        video_label.pack()
        self.video_entry = tk.Entry(self.root, width=50)
        self.video_entry.pack()
        video_button = tk.Button(self.root, text="Browse", command=self.select_video)
        video_button.pack()

        # Vehicle Type Selection
        self.vehicle_types = ['car', 'motorcycle', 'truck', 'bus', 'bicycle', 'person','auto-rickshaw']
        for vehicle in self.vehicle_types:
            var = tk.BooleanVar()
            cb = tk.Checkbutton(self.root, text=vehicle, variable=var)
            cb.pack(anchor='w')
            self.selected_vehicles[vehicle] = var

        # Day/Night Selection
        time_label = tk.Label(self.root, text="Select Time of Day:")
        time_label.pack()

        self.time_of_day = tk.StringVar(value="day")
        day_radio = tk.Radiobutton(self.root, text="Day", variable=self.time_of_day, value="day")
        night_radio = tk.Radiobutton(self.root, text="Night", variable=self.time_of_day, value="night")
        day_radio.pack(anchor='w')
        night_radio.pack(anchor='w')

        # Start Button
        start_button = tk.Button(self.root, text="Start Detection", command=self.start_detection)
        start_button.pack()

    def select_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("Video Files", "*.mp4;*.avi"), ("All Files", "*.*")))
        if file_path:
            self.video_entry.delete(0, tk.END)
            self.video_entry.insert(0, file_path)

    def start_detection(self):
        video_path = self.video_entry.get()

        if not video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return

        # Get selected vehicle types
        selected_types = [vehicle for vehicle, var in self.selected_vehicles.items() if var.get()]
        
        if not selected_types:
            messagebox.showerror("Error", "Please select at least one vehicle type to track.")
            return

        # Get selected time of day (day/night)
        time_of_day = self.time_of_day.get()

        # Call the detect_vehicles function here with selected vehicle types and time of day
        detect_vehicles(video_path, selected_types, time_of_day)

def main():
    root = tk.Tk()
    app = VehicleDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()