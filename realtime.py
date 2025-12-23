import cv2
import numpy as np
import joblib
from datetime import datetime
from insightface.app import FaceAnalysis


# ------------ CONFIG ------------
THRESH = 0.60           # recognition threshold
ABSENCE_GRACE = 2.0     # seconds: close interval if not seen for this long
LOG_FILE = "presence_log.csv"
# --------------------------------


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    clf = joblib.load("svm_face.pkl")
    names = np.load("names.npy", allow_pickle=True)

    app = FaceAnalysis(name="buffalo_l")
    # If you don't have GPU, set ctx_id=-1
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # active[label] = {"start": datetime, "last_seen": datetime, "max_conf": float}
    active = {}

    # Create log header if file doesn't exist
    try:
        with open(LOG_FILE, "x", encoding="utf-8") as f:
            f.write("label,start_time,end_time,duration_seconds,max_conf\n")
    except FileExistsError:
        pass

    def close_interval(label: str, end_time: datetime):
        """Close and write an active interval for label."""
        info = active.get(label)
        if not info:
            return
        start_time = info["start"]
        max_conf = info["max_conf"]
        duration = (end_time - start_time).total_seconds()

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"{label},{start_time.strftime('%Y-%m-%d %H:%M:%S')},"
                f"{end_time.strftime('%Y-%m-%d %H:%M:%S')},"
                f"{duration:.2f},{max_conf:.2f}\n"
            )
        del active[label]

    print(f"[{now_str()}] Logging to: {LOG_FILE}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = datetime.now()

        faces = app.get(frame)

        # labels seen in THIS frame
        seen_now = set()

        for face in faces:
            emb = face.embedding.astype(np.float32)
            emb = emb / np.linalg.norm(emb)

            probs = clf.predict_proba([emb])[0]
            best = int(np.argmax(probs))
            conf = float(probs[best])

            label = str(names[best]) if conf >= THRESH else "Unknown"
            seen_now.add(label)

            # update active intervals
            if label not in active:
                active[label] = {"start": t, "last_seen": t, "max_conf": conf}
            else:
                active[label]["last_seen"] = t
                if conf > active[label]["max_conf"]:
                    active[label]["max_conf"] = conf

            # draw bbox + label
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

        # Close intervals that have been absent for ABSENCE_GRACE seconds
        for label in list(active.keys()):
            last_seen = active[label]["last_seen"]
            if (t - last_seen).total_seconds() > ABSENCE_GRACE:
                close_interval(label, last_seen)

        cv2.imshow("Family Face Recognition (with log)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    # Close any still-active intervals at exit
    end_time = datetime.now()
    for label in list(active.keys()):
        close_interval(label, active[label]["last_seen"])

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{now_str()}] Done. Log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
