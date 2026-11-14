
import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Configure camera
cap.set(cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Camera not opened!")
    exit()

print("Webcam started successfully!")
print("Press Q to exit")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame capture failed")
        break

    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color (two HSV ranges)
    low_red1 = np.array([0, 100, 100])
    high_red1 = np.array([10, 255, 255])

    low_red2 = np.array([160, 100, 100])
    high_red2 = np.array([179, 255, 255])

    red_mask1 = cv2.inRange(hsv_frame, low_red1, high_red1)
    red_mask2 = cv2.inRange(hsv_frame, low_red2, high_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Blue color
    low_blue = np.array([90, 80, 50])
    high_blue = np.array([135, 255, 255])

    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)

    # Green color
    low_green = np.array([35, 50, 50])
    high_green = np.array([90, 255, 255])

    green_mask = cv2.inRange(hsv_frame, low_green, high_green)

    # Display detected colors
    red_result = cv2.bitwise_and(frame, frame, mask=red_mask)
    blue_result = cv2.bitwise_and(frame, frame, mask=blue_mask)
    green_result = cv2.bitwise_and(frame, frame, mask=green_mask)

    cv2.imshow("Original Webcam", frame)
    cv2.imshow("Red Detection", red_result)
    cv2.imshow("Blue Detection", blue_result)
    cv2.imshow("Green Detection", green_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
