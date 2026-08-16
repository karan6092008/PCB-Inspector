from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
app= FastAPI(title="PCB Defect Detection API", version="1.0")
try:
    model= YOLO("weights/best.pt")
except Exception:
    mode= None

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model is not None}
@app.post("/predict")    
async def predict_defects(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model weights missing in weights/best.pt ")
    contents= await file.read() 
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results= model(image, conf=0.4)[0]
    detections=[]
    counts={}
    for box in results.boxes:
        cls_id= int(box.cls[0])
        cls_name= model.names[cls_id]
        confidence=float(box.conf[0])
        bbox= [round(c, 1) for c in box.xyxy[0].tolist()]

        detection.append({
            "class_name": cls_name,
            "confidence": round(confidence, 2),
            "bounding_box":bbox
        })
        counts[cls_name] = counts.get(cls_name, 0) +1
    return{
        "filename": file.filename,
        "total_defects_found": len(detections),
        "summary": counts,
        "detections": detections
    }