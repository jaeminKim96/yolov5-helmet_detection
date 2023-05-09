import cv2
import torch
import numpy as np
from PIL import Image
from models.experimental import attempt_load
import pandas as pd

cap = cv2.VideoCapture(0)
width, height = 416, 416
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Load YOLOv5 model
model = attempt_load(
    "C:/Users/user/Desktop/yolov5-master/best.pt",
    device=torch.device("cpu"),  # gpu에서 학습한 걸 cpu에 로드할 때
)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    img = Image.fromarray(frame[..., ::-1])  # Convert BGR to RGB

    # Inference
    results = model(img, size=height)

    # Display result
    results.render()
    # results.pandas().xyxy[0]
    frame = np.array(img)[..., ::-1]  # Convert RGB to BGR
    cv2.imshow("YOLOv5", frame)

    if cv2.waitKey(1) == ord("q"):  # Quit when 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
