import cv2

# Load Haar Cascade XML file
face_classifier = cv2.CascadeClassifier(
    r"C:\Users\aisha\Downloads\haarcascade_frontalcatface.xml"
)

# Check classifier
if face_classifier.empty():
    print("Cascade file not loaded!")
    exit()

# Load image
img = cv2.imread(r"C:\Users\aisha\OneDrive\Documents\ME.jpeg")

if img is None:
    print("Image not loaded!")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(30, 30)
)

print("Faces detected:", len(faces))

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()