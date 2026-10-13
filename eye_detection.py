
import cv2
import mediapipe as mp
import numpy as np
import time

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    raise SystemExit(1)

# MediaPipe facial landmark detector
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices for the left and right eyes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.20
CLOSED_EYE_SECONDS = 1.0

closed_start = None

def calculate_ear(eye_points):
    # Vertical distances divided by horizontal eye width
    vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("Eye detection running. Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera.")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            left_points = np.array([
                [face.landmark[i].x * width,
                 face.landmark[i].y * height]
                for i in LEFT_EYE
            ], dtype=np.float64)

            right_points = np.array([
                [face.landmark[i].x * width,
                 face.landmark[i].y * height]
                for i in RIGHT_EYE
            ], dtype=np.float64)

            left_ear = calculate_ear(left_points)
            right_ear = calculate_ear(right_points)
            ear = (left_ear + right_ear) / 2.0

            # Mark the eye landmarks
            for point in np.vstack((left_points, right_points)):
                cv2.circle(
                    frame,
                    tuple(point.astype(int)),
                    2,
                    (0, 255, 0),
                    -1
                )

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if ear < EAR_THRESHOLD:
                if closed_start is None:
                    closed_start = time.monotonic()

                duration = time.monotonic() - closed_start

                if duration >= CLOSED_EYE_SECONDS:
                    status = "EYES CLOSED - WARNING!"
                    color = (0, 0, 255)
                else:
                    status = "Eyes closing..."
                    color = (0, 165, 255)
            else:
                closed_start = None
                duration = 0
                status = "Eyes Open"
                color = (0, 255, 0)

            cv2.putText(
                frame,
                status,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        else:
            closed_start = None
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Driver Drowsiness - Eye Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
