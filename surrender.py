import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    success,frame=cap.read()

    if success:
        RGB_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=hands.process(RGB_frame)

        if result.multi_hand_landmarks:

            for hands_landmarks in result.multi_hand_landmarks:

                print(hands_landmarks.landmark)

                mp_drawing.draw_landmarks(
                    frame,
                    hands_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            # Check if TWO hands are detected
            if len(result.multi_hand_landmarks) == 2:

                hands_raised = True

                # Check both hands
                for hands_landmarks in result.multi_hand_landmarks:

                    wrist = hands_landmarks.landmark[
                        mp_hands.HandLandmark.WRIST
                    ]

                    index = hands_landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ]

                    middle = hands_landmarks.landmark[
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                    ]

                    ring = hands_landmarks.landmark[
                        mp_hands.HandLandmark.RING_FINGER_TIP
                    ]

                    pinky = hands_landmarks.landmark[
                        mp_hands.HandLandmark.PINKY_TIP
                    ]

                    # If one hand is NOT raised
                    if not (
                        index.y < wrist.y and
                        middle.y < wrist.y and
                        ring.y < wrist.y and
                        pinky.y < wrist.y
                    ):
                        hands_raised = False

                # Both hands are raised
                if hands_raised:

                    cv2.putText(
                        frame,
                        "YOU ARE SURRENDERED!",
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()