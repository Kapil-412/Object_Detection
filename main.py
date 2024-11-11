import tkinter as tk
from tkinter import messagebox
import cv2
import os
from datetime import datetime
from PIL import Image, ImageTk
import pandas as pd
from ultralytics import YOLO  # Adjusted import for YOLOv8

# Load YOLOv8 model directly
model = YOLO('runs/detect/train/weights/best.pt')  # Directly load your model

# Create a directory to store captured images if it doesn't exist
capture_folder = "captured_images"
os.makedirs(capture_folder, exist_ok=True)

class BottleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bottle Detection and Counting")

        # IP camera setup
        self.ip_camera_url = 'http://<ip_address>:8080/video'  # Replace with actual IP address
        self.cap = cv2.VideoCapture(self.ip_camera_url)

        # Info panel
        self.info_label = tk.Label(root, text="Bottle Info: ", font=("Helvetica", 16))
        self.info_label.pack()

        # Status panel
        self.status_label = tk.Label(root, text="Bottles Placed: 0", font=("Helvetica", 14))
        self.status_label.pack()

        # Live feed panel
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Start button
        self.start_button = tk.Button(root, text="Start", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT)

        # End button
        self.end_button = tk.Button(root, text="End", command=self.end_detection)
        self.end_button.pack(side=tk.LEFT)

        # Capture button
        self.capture_button = tk.Button(root, text="Capture Image", command=self.capture_image, state=tk.DISABLED)
        self.capture_button.pack(side=tk.LEFT)

        self.running = False

    def start_detection(self):
        self.running = True
        self.capture_button.config(state=tk.DISABLED)  # Disable capture button during detection
        self.update_feed()

    def end_detection(self):
        self.running = False
        self.capture_button.config(state=tk.DISABLED)  # Disable capture button when not running
        if self.cap.isOpened():
            self.cap.release()  # Release video capture
        self.root.quit()  # Close the application gracefully

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            detected_bottles, results = self.detect_bottles(frame)  # Get detected bottles and results
            if detected_bottles:
                # Draw bounding boxes and total count on the frame
                self.annotate_frame(frame, results)

                # Create a unique ID and timestamp for the captured image
                unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = os.path.join(capture_folder, f"capture_{unique_id}.jpg")
                cv2.imwrite(image_filename, frame)  # Save the annotated image

                # Prepare data for Excel
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                total_count = len(detected_bottles)
                bottle_names = ', '.join(detected_bottles)

                # Save to Excel
                self.save_to_excel(image_filename, timestamp, total_count, bottle_names)

                messagebox.showinfo("Image Captured", f"Image saved as {image_filename}\nTotal Bottles: {total_count}\nBottles: {bottle_names}")

    def detect_bottles(self, frame):
        results = model.predict(frame)  # Use YOLOv8 model for prediction
        detected_bottles = []

        # Check results and extract bottle names
        for box in results[0].boxes:
            if box.conf > 0.5:  # Consider only boxes with a confidence greater than 0.5
                bottle_name = box.cls[0]  # Get the class name (index) for the detected bottle
                detected_bottles.append(model.names[int(bottle_name)])  # Get the bottle name from model names
        
        return detected_bottles, results  # Return both detected bottles and results

    def annotate_frame(self, frame, results):
        # Draw bounding boxes on the frame
        for box in results[0].boxes:
            if box.conf > 0.5:  # Consider only boxes with a confidence greater than 0.5
                # Get box coordinates and draw rectangle
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the bounding box coordinates
                bottle_name = model.names[int(box.cls[0])]  # Get the bottle name from model names
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, bottle_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Add total count text
        total_count = len([box for box in results[0].boxes if box.conf > 0.5])
        cv2.putText(frame, f'Total Bottles: {total_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def save_to_excel(self, image_filename, timestamp, total_count, bottle_names):
        # Create a DataFrame for the current capture data
        data = {
            "Image Filename": [image_filename],
            "Timestamp": [timestamp],
            "Total Count": [total_count],
            "Bottle Names": [bottle_names]
        }
        df = pd.DataFrame(data)

        # Define the Excel file name
        excel_file = 'bottle_detection_log.xlsx'
        
        # Check if the Excel file already exists
        if os.path.exists(excel_file):
            # Append new data below existing data
            with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                # Check if header exists
                existing_df = pd.read_excel(excel_file)
                if existing_df.empty:
                    df.to_excel(writer, index=False)  # Write header and data if file is empty
                else:
                    df.to_excel(writer, index=False, header=False, startrow=len(existing_df) + 1)  # Append data
        else:
            # Create a new Excel file with header and data
            df.to_excel(excel_file, index=False)

    def update_feed(self):
        if not self.running:
            return
        
        # Capture frame from IP camera
        ret, frame = self.cap.read()
        if ret:
            # Bottle detection
            detected_bottles, results = self.detect_bottles(frame)  # Get detected bottles and results
            detected_count = len(detected_bottles)  # Count detected bottles

            # Update the status label dynamically
            self.status_label.config(text=f"Bottles Placed: {detected_count}")

            # Annotate frame with bounding boxes and labels
            self.annotate_frame(frame, results)  # Pass results to the annotation method

            # Convert OpenCV image (BGR) to PIL format (RGB) and resize for display
            annotated_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(annotated_frame_rgb)
            img_pil = img_pil.resize((640, 480), Image.LANCZOS)  # Resize the image for display

            # Convert the PIL image to a format tkinter can handle
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update the canvas with the new image
            self.canvas.create_image(0, 0, anchor='nw', image=img_tk)
            self.canvas.image = img_tk  # Keep a reference to avoid garbage collection

            # Enable capture button if bottles are detected
            if detected_count > 0:  # You can adjust this condition based on your logic
                self.capture_button.config(state=tk.NORMAL)

        # Schedule the next frame update
        self.root.after(10, self.update_feed)

# Main function to run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = BottleDetectionApp(root)
    root.mainloop()
