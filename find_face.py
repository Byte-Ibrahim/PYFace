import face_recognition

image = face_recognition.load_image_file("Test3.jpg")
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

print(f"I found {len(face_encodings)} face(s) in this image.\n")

# Loop through every encoding found
for index, encoding in enumerate(face_encodings):
    print(f"--- Face #{index + 1} Digital Fingerprint (First 5 numbers) ---")
    print(encoding[:5]) 
    print("-" * 50)