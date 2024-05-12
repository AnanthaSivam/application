import cv2
import face_recognition
import numpy as np
import streamlit as st

# Load pre-trained face recognition model (consider accuracy and speed)
model = "vgg-face"  # Example: "vgg-face", "facenet", or custom model

# Known face encodings and names (replace with your data)
known_faces = []
known_names = []

def load_known_faces(known_faces_dir):
    for filename in os.listdir(known_faces_dir):
        img = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        face_encoding = face_recognition.face_encodings(img)[0]
        known_faces.append(face_encoding)
        known_names.append(os.path.splitext(filename)[0])

# Load known faces on app startup (optional for efficiency)
load_known_faces("known_faces")  # Replace with your known faces directory

def capture_video():
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam

    # Load face cascade classifier for basic face detection (optional, may not be needed for some models)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Path to cascade classifier file

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert frame to RGB format
        rgb_frame = frame[:, :, ::-1]

        # Resize frame for better performance (optional)
        # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Detect faces (using cascade classifier or model-specific method)
        faces = []
        if model in ["vgg-face", "facenet"]:  # Use model-specific face detection if applicable
            face_locations = face_recognition.face_locations(rgb_frame)
            faces = [(x, y, y + h, x + w) for (x, y, h, w) in face_locations]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Find encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, faces)

        # Check for matches
        for face_encoding, face in zip(face_encodings, faces):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            # Draw facial recognition bounding boxes and labels
            (x, y, bottom, right) = face
            cv2.rectangle(frame, (x, y), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the resulting frame
        st.image(frame, channels="BGR")

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture
    video_capture.release()
    cv2.destroyAllWindows()

# Main app functionality
if __name__ == "__main__":
    st.title("Face Recognition App")

    # Upload known faces (optional, allows user input)
    known_faces_uploaded = st.file_uploader("Upload Known Faces (Optional)", type=["jpg", "png"], accept_multiple=True)
    if known_faces_uploaded:
        for uploaded_file in known_faces_uploaded:
            bytes_data = uploaded_file.read()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)  # Decode uploaded image
            face_encoding = face_recognition.face_encodings(img
