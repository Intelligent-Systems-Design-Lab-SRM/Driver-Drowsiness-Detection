
import cv2
import mediapipe as mp
import numpy as np
import time

# Camera setup
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    raise SystemExit(1)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmark indices
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308

# Initial experimental thresholds
EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.20

EYE_CLOSED_SECONDS = 1.0
YAWN_SECONDS = 1.2

eye_closed_start = None
mouth_open_start = None


def calculate_ear(points):
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_landmark_points(face, indices, width, height):
    return np.array([
        [
            face.landmark[i].x * width,
            face.landmark[i].y * height
        ]
        for i in indices
    ], dtype=np.float64)


def calculate_mar(face, width, height):
    def point(index):
        landmark = face.landmark[index]
        return np.array([
            landmark.x * width,
            landmark.y * height
        ], dtype=np.float64)

    upper = point(UPPER_LIP)
    lower = point(LOWER_LIP)
    left = point(LEFT_MOUTH)
    right = point(RIGHT_MOUTH)

    mouth_height = np.linalg.norm(upper - lower)
    mouth_width = np.linalg.norm(right - left)

    if mouth_width == 0:
        return 0.0

    return mouth_height / mouth_width


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("Unified detection started. Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Camera frame unavailable.")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "NORMAL MONITORING"
        color = (0, 255, 0)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # Calculate EAR
            left_points = get_landmark_points(
                face, LEFT_EYE, width, height
            )
            right_points = get_landmark_points(
                face, RIGHT_EYE, width, height
            )

            left_ear = calculate_ear(left_points)
            right_ear = calculate_ear(right_points)
            ear = (left_ear + right_ear) / 2.0

            # Calculate MAR
            mar = calculate_mar(face, width, height)

            # Draw eye landmarks
            for point in np.vstack((left_points, right_points)):
                cv2.circle(
                    frame, tuple(point.astype(int)),
                    2, (0, 255, 0), -1
                )

            # Eye closure timer
            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.monotonic()
            else:
                eye_closed_start = None

            # Mouth opening timer
            if mar > MAR_THRESHOLD:
                if mouth_open_start is None:
                    mouth_open_start = time.monotonic()
            else:
                mouth_open_start = None

            eye_duration = (
                time.monotonic() - eye_closed_start
                if eye_closed_start is not None else 0
            )

            mouth_duration = (
                time.monotonic() - mouth_open_start
                if mouth_open_start is not None else 0
            )

            eye_warning = eye_duration >= EYE_CLOSED_SECONDS
            mouth_warning = mouth_duration >= YAWN_SECONDS

            # Combined decision
            if eye_warning and mouth_warning:
                status = "EYE CLOSURE + MOUTH WARNING"
                color = (0, 0, 255)
            elif eye_warning:
                status = "PROLONGED EYE CLOSURE!"
                color = (0, 0, 255)
            elif mouth_warning:
                status = "POSSIBLE YAWN!"
                color = (0, 165, 255)
            elif eye_closed_start is not None:
                status = "CHECKING EYE CLOSURE..."
                color = (0, 165, 255)
            elif mouth_open_start is not None:
                status = "CHECKING MOUTH OPENING..."
                color = (0, 165, 255)

            # Display measurements
            cv2.putText(
                frame, f"EAR: {ear:.3f}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"MAR: {mar:.3f}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

        else:
            eye_closed_start = None
            mouth_open_start = None
            status = "NO FACE DETECTED"
            color = (0, 0, 255)

        # Unified status display
        cv2.putText(
            frame, status,
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color, 2
        )

        cv2.imshow("Driver Drowsiness Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
