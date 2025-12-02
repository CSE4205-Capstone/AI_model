# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io
import os

app = FastAPI()

# =========================
# 1) YOLOv11 모델 로드
# =========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
model = YOLO(MODEL_PATH)


# =========================
# 2) Helper: 흔들림(Blur) 체크
# =========================
def is_blurry(cv_img, threshold=70.0):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, fm


# =========================
# 3) Helper: 밝기 체크
# =========================
def is_too_dark_or_bright(cv_img, low=40, high=220):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    return mean_val < low or mean_val > high, mean_val


@app.get("/")
def read_root():
    return {"status": "ok", "message": "YOLOv11 API server is running"}


# =========================
# 4) /detect 엔드포인트
# =========================
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    YOLOv11 추론 + 안전 처리(safe-guard) 포함
    """
    # 1) 파일 읽기
    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # PIL → CV2 변환
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    # ===========================
    # STEP A: 이미지 품질 체크
    # ===========================

    # A-1) 블러 체크
    blurry, fm = is_blurry(cv_img)
    if blurry:
        return JSONResponse({
            "status": "retry",
            "reason": "blurry_image",
            "detail": f"LaplacianVar={fm:.2f}"
        })

    # A-2) 밝기 체크
    bad_light, mean_val = is_too_dark_or_bright(cv_img)
    if bad_light:
        return JSONResponse({
            "status": "retry",
            "reason": "bad_light",
            "detail": f"brightness={mean_val:.2f}"
        })


    # ===========================
    # STEP B: YOLO 추론
    # ===========================
    results = model.predict(
        pil_img,
        imgsz=800,
        conf=0.6    # low로 둔 뒤 자체적으로 판단
    )

    r = results[0]

    boxes = r.boxes.xyxy.cpu().tolist()
    scores = r.boxes.conf.cpu().tolist()
    class_ids = r.boxes.cls.cpu().tolist()

    # 감지 없음 → 재촬영 요청
    if len(boxes) == 0:
        return JSONResponse({
            "status": "retry",
            "reason": "no_object_detected"
        })


    # ===========================
    # STEP C: 박스 기반 안정성 체크
    # ===========================

    detections = []
    final_class = None
    final_score = 0

    for box, score, cls_id in zip(boxes, scores, class_ids):
        xmin, ymin, xmax, ymax = box
        cls_id_int = int(cls_id)
        area_ratio = ((xmax - xmin) * (ymax - ymin)) / (w * h)

        # C-1) 너무 작은 물체? (너무 멀리)
        if area_ratio < 0.10:
            return JSONResponse({
                "status": "retry",
                "reason": "object_too_small",
                "area_ratio": float(area_ratio)
            })

        # 최고 conf 하나만 선택
        if score > final_score:
            final_score = score
            final_class = cls_id_int

        detections.append({
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "confidence": float(score),
            "class_id": cls_id_int,
            "class_name": r.names[cls_id_int],
            "area_ratio": float(area_ratio)
        })


    # ===========================
    # STEP D: confidence 안정성 판단
    # ===========================
    if final_score < 0.40:
        return JSONResponse({
            "status": "retry",
            "reason": "low_confidence",
            "confidence": float(final_score)
        })

    # ===========================
    # STEP E: 최종 정상 결과
    # ===========================
    return JSONResponse({
        "status": "ok",
        "pred_class": r.names[final_class],
        "class_id": final_class,
        "confidence": float(final_score),
        "detections": detections
    })
