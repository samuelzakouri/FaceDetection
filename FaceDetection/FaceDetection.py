import cv2
from time import sleep
import os
import subprocess
from datetime import date

# Path to directories
DATA_PATH = "../Data/"
FILTERS_PATH = "../Filters/"
SCREENSHOT_PATH = "../Screenshots_" + date.today().strftime("%b_%d_%Y") + '/'

# OpenCV Haar Cascade Classifiers
casc_face_Path = DATA_PATH + "haarcascade_frontalface_default.xml"  # for face detection
casc_eye_Path = DATA_PATH + "haarcascade_eye.xml"  # for eye detection

# if cv2 Cascade Classifiers are not downloaded
if not os.path.exists(casc_face_Path) or not os.path.exists(casc_eye_Path):
    subprocess.call([DATA_PATH + "download_filters.sh"])
else:
    print('Filters already exist!')

# cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier(casc_face_Path)
eyeCascade = cv2.CascadeClassifier(casc_eye_Path)

# Load filters
mustache_filter = cv2.imread(FILTERS_PATH + 'mustache.jpg')
hat_filter = cv2.imread(FILTERS_PATH + 'hat.png')

#Size of the screen
screen_width =1440
screen_height = 990

#Initialization
mustache = False
hat = False
blur = False

count = 1

# Open webcam on system
video_capture = cv2.VideoCapture(0)

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if ret:

        # Create greyscale image from the video feed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # Detect faces
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(60, 60)
        )

        # Detect eyes 
        #eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.35, minNeighbors=6)

        # For each faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            gray = gray[y:y + h, x:x + w]
            color = frame[y:y + h, x:x + w]

            if mustache:

                mustache_filter_width = int(w * 0.4166666) + 1
                mustache_filter_height = int(h * 0.142857) + 1

                mustache_filter = cv2.resize(mustache_filter, (mustache_filter_width, mustache_filter_height))

                for i in range(int(0.62857142857 * h), int(0.62857142857 * h) + mustache_filter_height):
                    for j in range(int(0.29166666666 * w), int(0.29166666666 * w) + mustache_filter_width):
                        for k in range(3):
                            if mustache_filter[i - int(0.62857142857 * h)][j - int(0.29166666666 * w)][k] < 235:
                                frame[y + i][x + j][k] = mustache_filter[i - int(0.62857142857 * h)][j - int(0.29166666666 * w)][k]
            elif hat:

                hat_width = w + 1
                hat_height = int(0.35 * h) + 1

                hat_filter = cv2.resize(hat_filter, (hat_width, hat_height))

                for i in range(hat_height):
                    for j in range(hat_width):
                        for k in range(3):
                            if hat_filter[i][j][k] < 235:
                                frame[y + i - int(0.25 * h)][x + j][k] = hat_filter[i][j][k]
            elif blur:
                # apply a gaussian blur
                color = cv2.GaussianBlur(color, (23, 23), 30)
                frame[y:y + color.shape[0], x:x + color.shape[1]] = color

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        name = 'Face Detection'
        # Display the resulting frame
        cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow(name, screen_width, screen_height)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        cv2.imshow(name, frame)

    # waitKey(0) will display the window infinitely until any keypress (it is suitable for image display).
    # waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
    ch = 0xFF & cv2.waitKey(1)

    # Activate/ Deactivate mustache filter
    if ch == ord("m"):
        mustache = not mustache

        if ch == ord('s'):
            # Create a directory screenshot with today's date
            if not os.path.exists(SCREENSHOT_PATH):
                os.makedirs(SCREENSHOT_PATH)
            name = "Screenshot" + str(count) + ".jpg"
            cv2.imwrite(SCREENSHOT_PATH + name, frame)
            count += 1

    # Activate/ Deactivate hat filter
    if ch == ord("h"):
        hat = not hat

        if ch == ord('s'):
            # Create a directory screenshot with today's date
            if not os.path.exists(SCREENSHOT_PATH):
                os.makedirs(SCREENSHOT_PATH)
            name = "Screenshot" + str(count) + ".jpg"
            cv2.imwrite(SCREENSHOT_PATH + name, frame)
            count += 1

    # Activate/ Deactivate Blur faces
    if ch == ord("b"):
        blur = not blur

        if ch == ord('s'):
            # Create a directory screenshot with today's date
            if not os.path.exists(SCREENSHOT_PATH):
                os.makedirs(SCREENSHOT_PATH)
            name = "Screenshot" + str(count) + ".jpg"
            cv2.imwrite(SCREENSHOT_PATH + name, frame)
            count += 1

    # Screenshot
    if ch == ord('s'):
        # Create a directory screenshot with today's date
        if not os.path.exists(SCREENSHOT_PATH):
            os.makedirs(SCREENSHOT_PATH)
        name = "Screenshot" + str(count) + ".jpg"
        cv2.imwrite(SCREENSHOT_PATH + name, frame)
        count += 1

    # Exit
    if ch == ord("q"):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
