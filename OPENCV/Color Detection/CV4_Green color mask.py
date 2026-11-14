
import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# Check camera
if not cap.isOpened():
    print("Error: Webcam not found!")
    exit()

print("Webcam started successfully!")
print("Press Q to exit")

while True:

    # Read webcam frame
    ret, frame = cap.read()

    # Check frame
    if not ret or frame is None:
        print("Error: Unable to capture frame!")
        break

    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green color range
    low_green = np.array([35, 100, 100])
    high_green = np.array([85, 255, 255])

    # Create green color mask
    green_mask = cv2.inRange(
        hsv_frame,
        low_green,
        high_green
    )

    # Extract green color
    green_result = cv2.bitwise_and(
        frame,
        frame,
        mask=green_mask
    )

    # Display output
    cv2.imshow("Original Webcam", frame)
    cv2.imshow("Green Color Mask", green_mask)
    cv2.imshow("Green Color Detection", green_result)

    # Exit using Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()




