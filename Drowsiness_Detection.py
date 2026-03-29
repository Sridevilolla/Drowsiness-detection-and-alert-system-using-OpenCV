from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import sys
import os

# ==============================
# Fix file paths for EXE
# ==============================
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==============================
# Initialize Alarm Sound
# ==============================
mixer.init()
mixer.music.load(resource_path("music.wav"))

# ==============================
# Eye Aspect Ratio Function
# ==============================
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ==============================
# Parameters
# ==============================
thresh = 0.25
frame_check = 40
flag = 0

# ==============================
# Load Face Detector + Predictor
# ==============================
print("Loading face detector...")
detect = dlib.get_frontal_face_detector()

print("Loading landmark model...")
predict = dlib.shape_predictor(
    resource_path("models/shape_predictor_68_face_landmarks.dat")
)

# Get eye landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# ==============================
# Start Camera
# ==============================
print("Starting camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
    exit()

# ==============================
# Main Loop
# ==============================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check drowsiness
        if ear < thresh:
            flag += 1

            if flag >= frame_check:
                cv2.putText(
                    frame,
                    "ALERT! DRIVER IS DROWSY!",
                    (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # Play alarm if not already playing
                if not mixer.music.get_busy():
                    mixer.music.play()

        else:
            flag = 0

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()
