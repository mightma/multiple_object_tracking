from ultralytics import YOLO

# Load a model
model = YOLO(
    "yolov10l.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(
    data="litter_1cls_dataset.yaml",
    epochs=300,
    imgsz=640,
    device=list(range(4)),  # 4 GPUs for ml.g5.12xlarge 
    name="litter_1cls_yolov10l",
    project="/opt/ml/model",
)
