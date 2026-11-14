
import cv2
import numpy as np

# Open webcam using DirectShow
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open webcam!")
    exit()

print("Webcam started successfully!")
print("Press Q to exit")

while True:

    ret, frame = cap.read()

    # Check frame capture
    if not ret or frame is None:
        print("Error: Cannot capture webcam frame!")
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White color range
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    # Create white mask
    white_mask = cv2.inRange(
        hsv, lower_white, upper_white
    )

    # Invert mask to detect every color except white
    mask = cv2.bitwise_not(white_mask)

    # Apply mask
    result = cv2.bitwise_and(
        frame, frame, mask=mask
    )

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Every Color Except White", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
