import os
import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # Path to your trained model

# Path to your dataset
image_folder = "NEWD"  # Adjust this if you want to use a different dataset folder
output_folder = "RESD"  # Folder to save the detection results

# Create output directory if it doesn't existt
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List of bottle classes (adjust these indices according to your model's labels)
BOTTLE_CLASSES = ['blackBottle', 'blackblueBottle', 'blueBigBottle', 'blueBottle', 'greenBottle', 'yellowBottle']
BOTTLE_CLASS_INDICES = [0, 1, 2, 3, 4, 5]  # Example class indices for the bottles

# Iterate over each image in the dataset
for image_file in os.listdir(image_folder):
    if image_file.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_folder, image_file)
        
        # Load the image
        image = cv2.imread(image_path)
        
        # Run YOLOv8 detection
        results = model(image)

        # Initialize the bottle count
        bottle_count = 0

        # Extract detected boxes and plot them on the image
        for result in results:  # Loop over results for different frames if needed
            boxes = result.boxes  # Get the boxes
            for box in boxes:
                if box.cls in BOTTLE_CLASS_INDICES:  # Check if the detected class is in the bottle classes
                    bottle_count += 1

            # Annotate the image with detection results
            annotated_image = result.plot()

            # Draw the bottle count on the image
            cv2.putText(annotated_image, f'Bottles: {bottle_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save the image with detection results
        output_path = os.path.join(output_folder, f"detection_{image_file}")
        cv2.imwrite(output_path, annotated_image)
        
        print(f"Processed and saved: {output_path} - Bottles counted: {bottle_count}")

print("Object detection on dataset images is complete.")
