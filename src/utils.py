import cv2
import numpy as np
from PIL import Image
def draw_custom_bboxes(image: Image.Image, detections: list) -> Image.Image:
    """
    Draws custom bounding boxes and labels onto a PIL image array.
    """
    img_np = np.array(image)
    for det in detections:
        bbox = det["bounding_box"]
        label = f"{det['class_name']} ({det['confidence']:.2f})"
        xmin, ymin, xmax, ymax =map(int, bbox)
        cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(
            img_np,
            label,
            (xmin, max(ymin -10, 15)),
            cv2.FONt_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )
    return Image.fromarray(img_np)