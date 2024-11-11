from ultralytics import YOLO

model = YOLO("yolov8x.pt")

model.train(data = "yolov8_config.yaml", epochs = 10, imgsz = 224, batch = 8, device = 'cpu')