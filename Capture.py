# Importing Library
import cv2

# Importing the FrontFace Trained Files
alg = "haarcascade_frontalface_default.xml"
haarCascade = cv2.CascadeClassifier(alg)

# Initialising Camera
cam = cv2.VideoCapture(0)

while True:

    # Reading Frame from camera
    _, img = cam.read()

    # Converting to grayscale to match the Trained Files
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the Dimensions of the Face in the current Frame
    face = haarCascade.detectMultiScale(grayImg, 1.3, 4)

    # Segregating X, Y, W, H from face and make rectangles for every captured face
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the current frame with rect
    cv2.imshow("Face Detection", img)

    # Quit the application if the Q key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release Camera and Destroy all windows
cam.release()
cv2.destroyAllWindows()
