import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from pathlib import Path

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Create data directory
DATA_DIR = "data"
Path(DATA_DIR).mkdir(exist_ok=True)

def extract_hand_landmarks(frame):
    """Extract hand landmarks from a frame"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    
    # Return fixed size (21 landmarks * 3 coordinates * 2 hands = 126 features)
    # If less than 2 hands, pad with zeros
    while len(landmarks) < 126:
        landmarks.append(0)
    
    return landmarks[:126]

def collect_sign_data(sign_name, num_samples=100):
    """Collect training data for a specific sign"""
    
    sign_dir = os.path.join(DATA_DIR, sign_name)
    Path(sign_dir).mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print(f"Collecting data for sign: {sign_name}")
    print(f"Press 'S' to start/stop collecting, 'Q' to quit")
    
    collecting = False
    sample_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Extract landmarks
        landmarks = extract_hand_landmarks(frame)
        
        # Draw frame info
        if collecting:
            cv2.putText(frame, f"Collecting: {sample_count}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save the landmarks
            if len(landmarks) > 0:
                file_path = os.path.join(sign_dir, f"{sign_name}_{sample_count}.pkl")
                with open(file_path, 'wb') as f:
                    pickle.dump(landmarks, f)
                sample_count += 1
                
                if sample_count >= num_samples:
                    print(f"Completed collecting {num_samples} samples for '{sign_name}'")
                    break
        else:
            cv2.putText(frame, "Press 'S' to start collecting", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, f"Sign: {sign_name}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("Collect Data", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            collecting = not collecting
            if collecting:
                sample_count = 0
    
    cap.release()
    cv2.destroyAllWindows()

def collect_alphabet(num_samples=100):
    """Collect A-Z alphabet automatically (press S to toggle per-letter capture)"""
    letters = [chr(c) for c in range(ord('A'), ord('Z')+1)]
    for letter in letters:
        print(f"\nPreparing to collect for letter '{letter}'.")
        input(f"Place your hand for '{letter}' then press Enter to begin capturing for this letter (or Ctrl+C to abort).")
        collect_sign_data(letter, num_samples)
        print(f"Finished collecting for '{letter}'.")

def main():
    print("=== Sign Language Data Collection ===")
    print("Options:")
    print("  1) Collect alphabet A-Z")
    print("  2) Collect custom signs (comma separated names)")
    choice = input("Choose [1/2]: ").strip() or "1"
    
    if choice == "1":
        num_samples_input = input("Number of samples per letter (default 100): ").strip()
        num_samples = int(num_samples_input) if num_samples_input else 100
        collect_alphabet(num_samples)
    else:
        print("Enter sign names separated by commas (e.g., hello,goodbye,thank_you)")
        signs_input = input("Enter sign names: ").strip()
        if not signs_input:
            print("No signs entered!")
            return
        signs = [s.strip() for s in signs_input.split(',')]
        num_samples_input = input("Number of samples per sign (default 100): ").strip()
        num_samples = int(num_samples_input) if num_samples_input else 100
        for sign in signs:
            collect_sign_data(sign, num_samples)
            print(f"Finished collecting '{sign}'")

if __name__ == "__main__":
    main()