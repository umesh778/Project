import cv2
import dlib
import numpy as np
import pyautogui

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Load the Haar cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier('D:/project/haarcascade_eye.xml')

# Set the screen size
screen_width, screen_height = pyautogui.size()

# Set the cursor speed
cursor_speed = 10





# Load the face detector and landmark predictor models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/project/shape_predictor_68_face_landmarks.dat")

# Define the indexes of the landmark points corresponding to the eye region
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

# Define the EAR threshold for detecting eye blinks
EAR_THRESHOLD = 0.2

# Define the number of consecutive frames the EAR is below the threshold to detect a blink
CONSECUTIVE_FRAMES_THRESHOLD = 3

# Initialize the frame counters and the blink status
frame_counter = 0
left_eye_blinking = False
right_eye_blinking = False

# Read the input video
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    
    
    
    
    
   
    
    
    
  # If eyes are detected, get the position of the first eye and map it to the cursor position
    if len(eyes) > 0:
        x, y, w, h = eyes[0]
        cursor_x = int((x + w/2) * screen_width / frame.shape[1])
        cursor_y = int((y + h/2) * screen_height / frame.shape[0])
        
        # Move the cursor to the mapped position
        pyautogui.moveTo(cursor_x, cursor_y, duration=0.00001*cursor_speed)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop over the detected faces
    for face in faces:
        # Get the facial landmarks for the face region
        landmarks = predictor(gray, face)

        # Get the landmark points for the left and right eye regions
        left_eye_points = np.array([(landmarks.part(index).x, landmarks.part(index).y) for index in LEFT_EYE_POINTS])
        right_eye_points = np.array([(landmarks.part(index).x, landmarks.part(index).y) for index in RIGHT_EYE_POINTS])
        for eye in [left_eye_points, right_eye_points]:
            eye_center = np.mean(eye, axis=0).astype(int)
            eye_radius = int(np.linalg.norm(eye[0] - eye[3]) / 2)
            cv2.circle(frame, tuple(eye_center), eye_radius, (0, 255, 0), 2)
        # Calculate the EAR for each eye
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)

        # Check if the EAR is below the threshold for the left eye
        if left_ear < EAR_THRESHOLD:
            if not left_eye_blinking:
                left_eye_blinking = True
                frame_counter = 0
            else:
                frame_counter += 1
                if frame_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
                    pyautogui.click()
                    print("Left eye blinking")
        else:
            left_eye_blinking = False

        # Check if the EAR is below the threshold for the right eye
        if right_ear < EAR_THRESHOLD:
            if not right_eye_blinking:
                right_eye_blinking = True
                frame_counter = 0
            else:
                if frame_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
                    pyautogui.rightClick()
                    print("right click")
        else:
            right_eye_blinking=False
    cv2.imshow('Eye Detection', frame)
    
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
            
