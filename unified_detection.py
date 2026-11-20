
import cv2
import mediapipe as mp
import numpy as np
import time

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


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera not found!")
        camera.release()
        return 1

    eye_closed_start = None
    mouth_open_start = None

    try:
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

                    left_points = get_landmark_points(
                        face, LEFT_EYE, width, height
                    )
                    right_points = get_landmark_points(
                        face, RIGHT_EYE, width, height
                    )

                    left_ear = calculate_ear(left_points)
                    right_ear = calculate_ear(right_points)
                    ear = (left_ear + right_ear) / 2.0
                    mar = calculate_mar(face, width, height)

                    for point in np.vstack((left_points, right_points)):
                        cv2.circle(
                            frame, tuple(point.astype(int)),
                            2, (0, 255, 0), -1
                        )

                    if ear < EAR_THRESHOLD:
                        if eye_closed_start is None:
                            eye_closed_start = time.monotonic()
                    else:
                        eye_closed_start = None

                    if mar > MAR_THRESHOLD:
                        if mouth_open_start is None:
                            mouth_open_start = time.monotonic()
                    else:
                        mouth_open_start = None

                    now = time.monotonic()
                    eye_duration = (
                        now - eye_closed_start
                        if eye_closed_start is not None else 0
                    )
                    mouth_duration = (
                        now - mouth_open_start
                        if mouth_open_start is not None else 0
                    )

                    eye_warning = eye_duration >= EYE_CLOSED_SECONDS
                    mouth_warning = mouth_duration >= YAWN_SECONDS

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

                cv2.putText(
                    frame, status,
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2
                )
                cv2.imshow("Driver Drowsiness Detection System", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
