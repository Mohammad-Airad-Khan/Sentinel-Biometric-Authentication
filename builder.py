import os
import shutil
import csv
import cv2
import numpy as np
import random

# --- PATHS ---
RAW_FACES = "raw_data/faces"  # Path to AT&T dataset (folders s1, s2...)
RAW_KEYS = "raw_data/DSL-StrongPasswordData.csv" # Path to CMU CSV
DEST_FACES = "dataset/faces"
DEST_KEYS = "dataset/keystrokes"

# Clean old data
if os.path.exists(DEST_FACES): shutil.rmtree(DEST_FACES)
if os.path.exists(DEST_KEYS): shutil.rmtree(DEST_KEYS)
os.makedirs(DEST_FACES)
os.makedirs(DEST_KEYS)

print("--- BUILDING CHIMERIC DATASET ---")

# 1. Process Keystrokes (CMU Dataset)
# The CMU dataset is one giant CSV. We need to split it by user.
print("Loading Keystrokes...")
keystroke_pool = {}

try:
    with open(RAW_KEYS, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            subject = row[0] # e.g., 's002'
            # CMU data format: [subject, session, rep, key1_down, key1_up...]
            # We only want the timings (starting from index 3)
            # We convert them to float
            timings = [float(x) for x in row[3:]]
            
            # CMU measures Hold-Time (H) and Up-Down-Time (UD). 
            # Your system uses Dwell and Flight. 
            # For this simulation, we will just take the first 18 values to match a "phrase".
            # Note: This is a simplification for the 'Chimeric' concept.
            sample = timings[:18] 
            
            if subject not in keystroke_pool:
                keystroke_pool[subject] = []
            keystroke_pool[subject].append(sample)
except FileNotFoundError:
    print(f"ERROR: Could not find {RAW_KEYS}. Please download the CMU dataset.")
    exit()

# 2. Process Faces (AT&T Dataset) & Fuse
print("Fusing Identities...")

face_folders = sorted([f for f in os.listdir(RAW_FACES) if f.startswith('s')])
keystroke_ids = sorted(list(keystroke_pool.keys()))

# Limit to the smaller of the two lists (usually 40 users from AT&T)
num_users = min(len(face_folders), len(keystroke_ids))

print(f"Creating {num_users} Chimeric Users...")

for i in range(num_users):
    # The 'Chimeric' Identity
    face_src_id = face_folders[i]     # e.g., s1
    key_src_id = keystroke_ids[i]     # e.g., s002
    
    new_user_name = f"User_{i+1:03d}"
    
    # A. Copy Faces
    src_face_dir = os.path.join(RAW_FACES, face_src_id)
    dst_face_dir = os.path.join(DEST_FACES, new_user_name)
    os.makedirs(dst_face_dir, exist_ok=True)
    
    for img_file in os.listdir(src_face_dir):
        if img_file.endswith(".pgm"): # AT&T uses .pgm
            # Convert .pgm to .jpg for your system
            img = cv2.imread(os.path.join(src_face_dir, img_file), cv2.IMREAD_GRAYSCALE)
            new_name = img_file.replace(".pgm", ".jpg")
            cv2.imwrite(os.path.join(dst_face_dir, new_name), img)

    # B. Create Keystroke CSV
    # Your system expects a CSV where each row is a sample
    user_samples = keystroke_pool[key_src_id]
    # We use 50% for training (Enrollment) and 50% will be used for testing later
    # But for the dataset folder, we just dump them all.
    
    key_file = os.path.join(DEST_KEYS, f"{new_user_name}.csv")
    with open(key_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(user_samples)
        
    print(f"  [+] Created {new_user_name} (Face: {face_src_id} + Keys: {key_src_id})")

print("\nDONE! You now have a dataset of 40 'Virtual' users.")
