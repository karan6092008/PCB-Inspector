from roboflow import Roboflow
rf = Roboflow(api_key="PUBLIC_VS_CODE") # Public read
project = rf.workspace("vrb").project("pcb-defect-detection-0lyiv")
dataset = project.version(1).download("yolov8", location="data")