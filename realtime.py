import cv2, numpy as np, joblib
from insightface.app import FaceAnalysis

clf = joblib.load("svm_face.pkl")
names = np.load("names.npy", allow_pickle=True)

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # GPU yoxdursa ctx_id=-1

THRESH = 0.60  # bacının şəkli azdır deyə bir az aşağı

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    faces = app.get(frame)
    for face in faces:
        emb = face.embedding.astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        probs = clf.predict_proba([emb])[0]
        best = int(np.argmax(probs))
        conf = float(probs[best])

        label = str(names[best]) if conf >= THRESH else "Unknown"

        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Family Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
