import cv2
import face_recognition
import os
import csv
from datetime import datetime

# Path to student images
path = 'students'

images = []
student_names = []

# Load images
for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    images.append(img)

    # Remove .jpg extension
    student_names.append(os.path.splitext(file)[0])

print(student_names)

# Encode faces
def encode_faces(images):
    encoded_list = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encoded_list.append(encode)

    return encoded_list

# Mark attendance
def mark_attendance(name):
    with open('attendance.csv', 'a+', newline='') as f:
        writer = csv.writer(f)

        f.seek(0)
        data = f.readlines()

        name_list = []

        for line in data:
            entry = line.split(',')
            name_list.append(entry[0])

        if name not in name_list:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')

            writer.writerow([name, time])

print("Encoding faces...")
known_encodings = encode_faces(images)
print("Encoding complete.")

# Start webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    # Resize frame
    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    # Convert color
    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )

    # Find faces
    face_locations = face_recognition.face_locations(
        rgb_small_frame
    )

    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations
    )

    # Compare faces
    for encode_face, face_location in zip(
        face_encodings,
        face_locations
    ):

        matches = face_recognition.compare_faces(
            known_encodings,
            encode_face
        )

        face_distance = face_recognition.face_distance(
            known_encodings,
            encode_face
        )

        match_index = face_distance.argmin()

        if matches[match_index]:

            name = student_names[match_index].upper()

            y1, x2, y2, x1 = face_location

            # Scale back up
            y1, x2, y2, x1 = (
                y1 * 4,
                x2 * 4,
                y2 * 4,
                x1 * 4
            )

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Name box
            cv2.rectangle(
                frame,
                (x1, y2 - 35),
                (x2, y2),
                (0, 255, 0),
                cv2.FILLED
            )

            cv2.putText(
                frame,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            mark_attendance(name)

    cv2.imshow('AI Attendance System', frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()