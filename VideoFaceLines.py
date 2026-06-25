import cv2
import face_recognition

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("Starting Sci-Fi Mesh Camera... Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find all facial landmarks in the frame
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    for face_landmarks in face_landmarks_list:
        # Each 'face_landmarks' dictionary contains keys like 'chin', 'left_eyebrow', etc.
        # We loop through each feature to draw lines connecting the points
        for facial_feature in face_landmarks.keys():
            points = face_landmarks[facial_feature]
            
            # Loop through the points of this specific feature and draw lines between them
            for i in range(len(points) - 1):
                pt1 = points[i]
                pt2 = points[i+1]
                
                # Draw a neon white line connecting point 1 to point 2
                # Parameters: (image, point1, point2, color_in_BGR, thickness)
                cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
            
            # Optional: Draw small dots on every single landmark point to look like nodes
            for point in points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1) # Green dots

    # Display the result
    cv2.imshow('Cyberpunk Facial Mesh', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()