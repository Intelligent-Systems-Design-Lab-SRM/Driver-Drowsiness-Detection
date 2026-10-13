
import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    raise SystemExit(1)

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("Face detection running! Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            height, width, _ = frame.shape

            for point in face.landmark:
                x = int(point.x * width)
                y = int(point.y * height)

                cv2.circle(
                    frame, (x, y), 1, (0, 255, 0), -1
                )

            cv2.putText(
                frame, "FACE DETECTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                frame, "NO FACE DETECTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2
            )

        cv2.imshow("Facial Landmark Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
