from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='dataset.yaml', epochs=100, imgsz=640, workers=0)#, device=0)