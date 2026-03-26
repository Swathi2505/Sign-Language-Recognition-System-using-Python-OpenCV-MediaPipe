import cv2
import mediapipe as mp
import numpy as np
import pickle
import joblib
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load model and sign names
MODEL_PATH = "sign_language_model.pkl"
NAMES_PATH = "sign_names.pkl"

def load_model():
    """Load trained model and sign names"""
    try:
        model = joblib.load(MODEL_PATH)
        with open(NAMES_PATH, 'rb') as f:
            sign_names = pickle.load(f)
        print("Model loaded successfully!")
        return model, sign_names
    except FileNotFoundError:
        print(f"Error: Model file not found. Please train the model first using train_model.py")
        return None, None

def extract_hand_landmarks(frame):
    """Extract hand landmarks from a frame"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    landmarks = []
    hand_detected = False
    
    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    
    # Return fixed size (21 landmarks * 3 coordinates * 2 hands = 126 features)
    while len(landmarks) < 126:
        landmarks.append(0)
    
    return landmarks[:126], hand_detected

def predict_sign(model, landmarks, confidence_threshold=0.3):
    """Predict sign from landmarks"""
    landmarks = np.array(landmarks).reshape(1, -1)
    
    # Get prediction and confidence
    prediction = model.predict(landmarks)[0]
    probabilities = model.predict_proba(landmarks)[0]
    confidence = np.max(probabilities)
    
    return prediction, confidence

def main():
    model, sign_names = load_model()
    
    if model is None:
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("=== Real-Time Sign Language Recognition ===")
    print("Press 'Q' to quit")
    print("Press 'R' to reset smoothing buffer")
    
    # Smoothing: use last N predictions to smooth output
    prediction_buffer = deque(maxlen=7)
    
    # Word/sentence assembly state
    current_word = ""
    sentence_words = []
    last_confirmed_idx = None
    no_hand_frames = 0
    NO_HAND_THRESHOLD = 25  # frames of no hand to commit current word
    CONFIRM_COUNT = 4       # number of consistent frames to confirm a letter
    CONF_THRESH = 0.35      # min avg confidence to accept a letter
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Extract landmarks
        landmarks, hand_detected = extract_hand_landmarks(frame)
        
        # Display bounding rectangle
        cv2.rectangle(frame, (0, 0), (w, h), (200, 200, 200), 2)
        
        if hand_detected:
            no_hand_frames = 0
            # Predict
            prediction_idx, confidence = predict_sign(model, landmarks)
            
            if confidence > CONF_THRESH:
                prediction_buffer.append((prediction_idx, confidence))
            
            # Try to confirm a stable letter
            if len(prediction_buffer) >= CONFIRM_COUNT:
                preds = [p[0] for p in prediction_buffer]
                confs = [p[1] for p in prediction_buffer]
                from collections import Counter
                most_common_pred, count = Counter(preds).most_common(1)[0]
                avg_conf = np.mean(confs)
                
                # Confirm only if stable and high average confidence
                if count >= CONFIRM_COUNT and avg_conf >= CONF_THRESH and most_common_pred != last_confirmed_idx:
                    letter = sign_names[most_common_pred]
                    # Append confirmed letter (expect sign_names to be single letters for alphabet)
                    current_word += str(letter)
                    last_confirmed_idx = most_common_pred
                    prediction_buffer.clear()  # avoid duplicates
        else:
            # No hand detected: increment counter and commit word if timeout
            no_hand_frames += 1
            if no_hand_frames > NO_HAND_THRESHOLD and current_word:
                sentence_words.append(current_word)
                current_word = ""
                last_confirmed_idx = None
                prediction_buffer.clear()
        
        # Show assembled sentence and current word
        sentence_text = " ".join(sentence_words)
        cv2.putText(frame, f"Sentence: {sentence_text}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 0), 2)
        cv2.putText(frame, f"Word: {current_word}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Show available signs (for debugging)
        cv2.putText(frame, "Available signs:", (w - 300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        for i, name in enumerate(sign_names):
            cv2.putText(frame, f"- {name}", (w - 300, 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Sign Language Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            prediction_buffer.clear()
            current_word = ""
            sentence_words = []
            last_confirmed_idx = None
            no_hand_frames = 0
            print("Buffer and text cleared")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()