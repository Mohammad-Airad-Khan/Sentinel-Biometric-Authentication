import os
import cv2
import numpy as np
import joblib
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

FACE_DIR = "dataset/faces"
KEY_DIR = "dataset/keystrokes"

# --- NEW PREPROCESSING FUNCTION ---
def apply_clahe(gray_img):
    """Normalizes lighting to prevent 0.00 scores caused by shadows."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_img)

print("--- STARTING EVALUATION ---")

users = sorted(os.listdir(FACE_DIR))
if not users:
    print("No data found! Run the builder script first.")
    exit()

# 1. PREPARE DATA
print(f"Found {len(users)} Chimeric Users.")
X_faces_train, y_faces_train = [], []
X_faces_test, y_faces_test = [], []

X_keys_train, y_keys_train = [], []
X_keys_test, y_keys_test = [], []

label_map = {u: i for i, u in enumerate(users)}

for user in users:
    uid = label_map[user]
    
    # -- Load Faces --
    user_face_path = os.path.join(FACE_DIR, user)
    images = sorted(os.listdir(user_face_path))
    
    # Split: 5 for training, 5 for testing (AT&T has 10 per user)
    for i, img_name in enumerate(images):
        img_path = os.path.join(user_face_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # === APPLY CLAHE HERE FOR CONSISTENCY ===
        img = apply_clahe(img)
        
        img = cv2.resize(img, (200, 200)) # Ensure standard size
        
        if i < 5:
            X_faces_train.append(img)
            y_faces_train.append(uid)
        else:
            X_faces_test.append(img)
            y_faces_test.append(uid)

    # -- Load Keystrokes --
    user_key_path = os.path.join(KEY_DIR, f"{user}.csv")
    with open(user_key_path, 'r') as f:
        reader = csv.reader(f)
        samples = list(reader)
        samples = [list(map(float, row)) for row in samples if row]
        
        split_idx = len(samples) // 2
        for i, sample in enumerate(samples):
            if i < split_idx:
                X_keys_train.append(sample)
                y_keys_train.append(uid)
            else:
                X_keys_test.append(sample)
                y_keys_test.append(uid)

# 2. TRAIN MODELS
print("Training Face Model...")
face_model = cv2.face.LBPHFaceRecognizer_create()
face_model.train(X_faces_train, np.array(y_faces_train))

# Save the model for use in the Live System
if not os.path.exists("models"):
    os.makedirs("models")
face_model.save("models/face_model.yml")

print("Training Keystroke Model...")
# --- FIX: USE GLOBAL PADDING TO PREVENT INHOMOGENEOUS SHAPE ERROR ---
# Find the absolute longest sequence in both train and test sets
all_samples = X_keys_train + X_keys_test
max_len = max(len(x) for x in all_samples)

# Pad every single sample to that specific length
X_keys_train = [x + [0]*(max_len-len(x)) for x in X_keys_train]
X_keys_test = [x + [0]*(max_len-len(x)) for x in X_keys_test]

key_model = RandomForestClassifier(n_estimators=100)
key_model.fit(X_keys_train, y_keys_train)

# 3. TEST & EVALUATE
print("\n--- RESULTS ---")

# Evaluate Faces
correct_faces = 0
total_faces = 0
for i, img in enumerate(X_faces_test):
    label, conf = face_model.predict(img)
    true_label = y_faces_test[i]
    
    # Strictly for Identity (Who are you?)
    if label == true_label:
        correct_faces += 1
    total_faces += 1

print(f"Face Recognition Accuracy (with CLAHE): {correct_faces / total_faces * 100:.2f}%")

# Evaluate Keystrokes
key_preds = key_model.predict(X_keys_test)
key_acc = accuracy_score(y_keys_test, key_preds)
print(f"Keystroke Recognition Accuracy: {key_acc * 100:.2f}%")

# Multimodal Fusion Logic
print("\nMultimodal Simulation (Fusion Rule: AND):")
combined_correct = 0
combined_total = min(len(X_faces_test), len(X_keys_test))

for i in range(combined_total):
    f_label, f_conf = face_model.predict(X_faces_test[i])
    f_match = (f_label == y_faces_test[i])
    
    k_label = key_model.predict([X_keys_test[i]])[0]
    k_match = (k_label == y_keys_test[i])
    
    if f_match and k_match:
        combined_correct += 1

print(f"Total Combined Accuracy: {combined_correct / combined_total * 100:.2f}%")
