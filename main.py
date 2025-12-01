# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI()

# =========================
# 1) YOLOv11 모델 로드
# =========================
# (1) 네가 직접 학습한 모델이 있다면:
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
model = YOLO(MODEL_PATH)

# (2) 만약 그냥 공식 yolo11n 쓰고 싶으면:
# model = YOLO("yolo11n.pt")   # 자동으로 다운로드


@app.get("/")
def read_root():
    return {"status": "ok", "message": "YOLOv11 API server is running"}


# =========================
# 2) /detect 엔드포인트
# =========================
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    앱에서 이미지를 업로드하면,
    YOLOv11로 추론하고 결과를 JSON으로 리턴
    """
    # 1) 파일 읽기
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 2) YOLOv11 추론
    #    results: list, 여기서는 단일 이미지라서 results[0]만 사용
    results = model.predict(
        img,
        imgsz=800,   # 네가 학습할 때 쓴 사이즈에 맞춰주면 좋음
        conf=0.5    # confidence threshold 필요하면 조절
    )

    r = results[0]

    detections = []
    # r.boxes.xyxy: [N, 4], r.boxes.conf: [N], r.boxes.cls: [N]
    boxes = r.boxes.xyxy.cpu().tolist()
    scores = r.boxes.conf.cpu().tolist()
    class_ids = r.boxes.cls.cpu().tolist()

    for box, score, cls_id in zip(boxes, scores, class_ids):
        xmin, ymin, xmax, ymax = box
        cls_id_int = int(cls_id)
        detections.append({
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "confidence": float(score),
            "class_id": cls_id_int,
            "class_name": r.names[cls_id_int]
        })

    return JSONResponse(content={"detections": detections})
