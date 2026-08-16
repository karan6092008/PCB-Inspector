import shutil 
from pathlib import Path
from ultralytics import YOLO
def train_and_export():
    model = YOLO("yolov8n.pt")
    print("Starting YOLOv8 Fine Tunning....")
    results = model.train(
        data="data/data.yaml",
        epochs=50,
        patience=10,
        imgsz=640,
        batch=16,
        project="runs",
        name="pcb_inspection"
    )
    
    best_weights= Path("runs/pcb_inspection/weights/best.pt")
    target_weights= Path("weights/best.pt")
    
    if best_weights.exists():
        target_weights.parent.mkdir(exist_ok=True)
        shutil.copy(best_weights, target_weights)
        print(f"Success! Trained model exported to: {target_weights.resolve()}")
if __name__ == "__main__":
    train_and_export()

