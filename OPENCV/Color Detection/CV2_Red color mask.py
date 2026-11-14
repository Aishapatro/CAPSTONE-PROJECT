
import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check webcam
if not cap.isOpened():
    print("Error: Webcam not detected!")
    exit()

while True:

    ret, frame = cap.read()

    # Check whether frame is captured
    if not ret or frame is None:
        print("Error: Cannot read webcam frame!")
        break

    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color lower and upper ranges
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])

    # Create mask
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)

    # Apply mask
    red_result = cv2.bitwise_and(
        frame, frame, mask=red_mask
    )

    # Display windows
    cv2.imshow("Original Webcam", frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Red Color Detection", red_result)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()
