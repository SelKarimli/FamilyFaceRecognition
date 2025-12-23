# label_image.py
# Usage:
#   python label_image.py --input input.jpg --output output.jpg --thresh 0.60

import argparse
import os
import cv2
import numpy as np
import joblib
from insightface.app import FaceAnalysis

def draw_label_box(img, bbox, text, color=(0, 255, 0), thickness=3, font_scale=1.2):
    x1, y1, x2, y2 = bbox

    # Square-style box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Label on top-left, just like your screenshot
    # Put text slightly above the box if possible
    y_text = y1 - 12 if y1 - 12 > 15 else y1 + 28
    cv2.putText(
        img,
        text,
        (x1, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image (jpg/png)")
    parser.add_argument("--output", default="labeled_output.jpg", help="Path to save output jpg")
    parser.add_argument("--thresh", type=float, default=0.60, help="Unknown threshold (0.55-0.70 typical)")
    parser.add_argument("--ctx", type=int, default=0, help="GPU=0, CPU=-1")
    args = parser.parse_args()

    # Load model + class names
    if not os.path.exists("svm_face.pkl") or not os.path.exists("names.npy"):
        raise FileNotFoundError("Missing svm_face.pkl or names.npy in current folder.")

    clf = joblib.load("svm_face.pkl")
    names = np.load("names.npy", allow_pickle=True)

    # Load image
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    # Face detector + embedding extractor (ArcFace pack)
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=args.ctx, det_size=(640, 640))

    faces = app.get(img)

    for face in faces:
        emb = face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        probs = clf.predict_proba([emb])[0]
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])

        label = str(names[best_idx]) if conf >= args.thresh else "Unknown"
        text = f"{label} {conf:.2f}"

        x1, y1, x2, y2 = face.bbox.astype(int)
        draw_label_box(img, (x1, y1, x2, y2), text)

    # Save as JPG
    cv2.imwrite(args.output, img)
    print(f"Saved: {args.output} (faces: {len(faces)})")

if __name__ == "__main__":
    main()
