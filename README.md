Created a custom dataset of face images for training. The images are not uploaded as it is a large data set which requires huge memory. 
Utilized Haar Cascade Classifier to detect and crop eye images from the face dataset. 
Developed and trained a deep learning model on the cropped eye images as shown in ModelForDetectingEyeMoments.ipynb
Implemented real-time cursor movement control using the trained model. 
Plotted six points around the eye and calculated the Eye Aspect Ratio (EAR) for click detection as shown in eye_cursor.ipynb
Achieved left and right mouse clicks based on EAR for right and left eye blinks 
