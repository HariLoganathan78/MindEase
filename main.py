import cv2
import time
from deepface import DeepFace

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Set delay in seconds
last_update_time = 0
update_interval = 2  # seconds
latest_result = None

while True:
    ret, frame = cap.read()

    current_time = time.time()

    if current_time - last_update_time > update_interval:
        try:
            # Analyze the current frame for emotion
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                dominant_emotion = first_result['dominant_emotion']
                emotion_scores = first_result['emotion']

                # Store latest result to avoid frequent updates
                latest_result = {
                    "dominant_emotion": dominant_emotion,
                    "emotion_scores": emotion_scores
                }

            last_update_time = current_time

        except Exception as e:
            print(f"Error in DeepFace analysis: {e}")

    # If we have a recent result, draw it
    if latest_result:
        dominant_emotion = latest_result["dominant_emotion"]
        emotion_scores = latest_result["emotion_scores"]

        # Emotions to display
        display_emotions = ['happy', 'neutral', 'surprise', 'sad', 'angry']

        # Title
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Scores
        for i, emotion in enumerate(display_emotions):
            if emotion in emotion_scores:
                text = f"{emotion}: {emotion_scores[emotion]:.2f}%"
                cv2.putText(frame, text, (50, 100 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the result
    cv2.imshow('Emotion Detector', frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
