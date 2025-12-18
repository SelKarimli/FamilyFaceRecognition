import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X = np.load("X.npy")
y = np.load("y.npy")
names = np.load("names.npy", allow_pickle=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=names))

joblib.dump(clf, "svm_face.pkl")
print("Saved svm_face.pkl")
