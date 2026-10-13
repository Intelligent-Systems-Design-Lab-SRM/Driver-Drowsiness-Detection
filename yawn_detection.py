
import cv2
import mediapipe as mp
import numpy as np
import time

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    raise SystemExit(1)

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe mouth landmarks
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308

# Initial experimental settings; calibrate for your face
MAR_THRESHOLD = 0.20
YAWN_SECONDS = 1.2

yawn_start = None


def calculate_mar(face, width, height):
    def point(index):
        landmark = face.landmark[index]
        return np.array(
            [landmark.x * width, landmark.y * height],
            dtype=np.float64
        )

    upper = point(UPPER_LIP)
    lower = point(LOWER_LIP)
    left = point(LEFT_MOUTH)
    right = point(RIGHT_MOUTH)

    mouth_opening = np.linalg.norm(upper - lower)
    mouth_width = np.linalg.norm(left - right)

    if mouth_width == 0:
        return 0.0

    return mouth_opening / mouth_width


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("Yawning detection running. Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            mar = calculate_mar(face, width, height)

            # Display the mouth landmarks
            for index in [
                UPPER_LIP, LOWER_LIP, LEFT_MOUTH, RIGHT_MOUTH
            ]:
                landmark = face.landmark[index]
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame, (x, y), 4, (0, 255, 0), -1
                )

            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            if mar > MAR_THRESHOLD:
                if yawn_start is None:
                    yawn_start = time.monotonic()

                duration = time.monotonic() - yawn_start

                if duration >= YAWN_SECONDS:
                    status = "POSSIBLE YAWN DETECTED"
                    color = (0, 0, 255)
                else:
                    status = "Mouth open - checking..."
                    color = (0, 165, 255)
            else:
                yawn_start = None
                status = "Mouth normal"
                color = (0, 255, 0)

            cv2.putText(
                frame,
                status,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )

        else:
            yawn_start = None
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2
            )

        cv2.imshow("Driver Drowsiness - Yawn Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
