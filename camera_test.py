
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
    raise SystemExit(1)

print("Camera working! Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
