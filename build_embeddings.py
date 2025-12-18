import os, cv2, numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

DATA_DIR = "dataset_raw"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))#GPU yoxdur, ctx_id=-1

X, y = [], []
names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

for label, person in enumerate(names):
    person_dir = os.path.join(DATA_DIR, person)
    imgs = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]

    for f in tqdm(imgs, desc=f"{person}"):
        path = os.path.join(person_dir, f)
        img = cv2.imread(path)
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        face = max(faces, key=lambda fa: (fa.bbox[2]-fa.bbox[0])*(fa.bbox[3]-fa.bbox[1]))
        emb = face.embedding.astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        X.append(emb)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
names = np.array(names, dtype=object)

np.save("X.npy", X)
np.save("y.npy", y)
np.save("names.npy", names)

print("DONE")
print("X:", X.shape, "y:", y.shape)
print("People:", list(names))
