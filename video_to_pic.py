import cv2, os
from insightface.app import FaceAnalysis

VIDEO_PATH = "Ana.mp4"
OUT_DIR = "dataset_raw/Ana"
EVERY_N_FRAMES = 10  # təxminən 1 fps kimi (videonun fps-indən asılıdır)

os.makedirs(OUT_DIR, exist_ok=True)

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # GPU yoxdursa ctx_id=-1

cap = cv2.VideoCapture(VIDEO_PATH)

i, saved = 0, 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if i % EVERY_N_FRAMES == 0:
        faces = app.get(frame)
        if len(faces) > 0:
            saved += 1
            cv2.imwrite(os.path.join(OUT_DIR, f"face_{saved:04d}.jpg"), frame)

    i += 1

cap.release()
print("Saved face-frames:", saved)
