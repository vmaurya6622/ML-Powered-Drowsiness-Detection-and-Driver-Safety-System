# import cv2
# import mediapipe as mp

# # Initialize Mediapipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# # Define drawing specifications
# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# # Start video capture
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     raise IOError("Cannot open webcam")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the frame to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame for face landmarks
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Get coordinates of the left and right eyes
#             # Left eye landmarks (Refer Mediapipe's landmark indices)
#             left_eye = [face_landmarks.landmark[i] for i in range(133, 144)]
#             right_eye = [face_landmarks.landmark[i] for i in range(362, 373)]

#             # Convert normalized coordinates to pixel coordinates
#             h, w, _ = frame.shape
#             left_eye_coords = [(int(pt.x * w), int(pt.y * h)) for pt in left_eye]
#             right_eye_coords = [(int(pt.x * w), int(pt.y * h)) for pt in right_eye]

#             # Draw eye contours on the frame
#             for coord in left_eye_coords:
#                 cv2.circle(frame, coord, 1, (0, 255, 0), -1)
#             for coord in right_eye_coords:
#                 cv2.circle(frame, coord, 1, (0, 255, 0), -1)

#             # Extract eye regions (bounding boxes)
#             def extract_eye_region(eye_coords):
#                 x_min = min([x for x, y in eye_coords])
#                 x_max = max([x for x, y in eye_coords])
#                 y_min = min([y for x, y in eye_coords])
#                 y_max = max([y for x, y in eye_coords])
#                 return frame[y_min:y_max, x_min:x_max]

#             left_eye_roi = extract_eye_region(left_eye_coords)
#             right_eye_roi = extract_eye_region(right_eye_coords)

#             # Display extracted eyes (Optional)
#             if left_eye_roi.size > 0:
#                 cv2.imshow("Left Eye", cv2.resize(left_eye_roi, (100, 50)))
#             if right_eye_roi.size > 0:
#                 cv2.imshow("Right Eye", cv2.resize(right_eye_roi, (100, 50)))

#     # Show the frame with eye landmarks
#     cv2.imshow('Eye Detection', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import numpy as np
# import winsound
# import tensorflow as tf

# # Initialize Mediapipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Define beep parameters
# frequency = 2500
# duration = 1000

# # Load your pre-trained model
# new_model = tf.keras.models.load_model('my_model.keras')  # Replace with your model path

# # Initialize variables
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")

# counter = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the frame to RGB (required by Mediapipe)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)

#     status = "Face Not Detected"

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Extract eye landmarks (use indices from Mediapipe Face Mesh)
#             left_eye_indices = [33, 160, 158, 133, 153, 144]
#             right_eye_indices = [362, 385, 387, 263, 373, 380]

#             h, w, _ = frame.shape

#             # Get bounding boxes for left and right eyes
#             left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
#             right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]

#             # Draw eye regions on the frame
#             for point in left_eye + right_eye:
#                 cv2.circle(frame, point, 2, (0, 255, 0), -1)

#             # Crop eye regions
#             x_min_left = min([p[0] for p in left_eye])
#             y_min_left = min([p[1] for p in left_eye])
#             x_max_left = max([p[0] for p in left_eye])
#             y_max_left = max([p[1] for p in left_eye])

#             x_min_right = min([p[0] for p in right_eye])
#             y_min_right = min([p[1] for p in right_eye])
#             x_max_right = max([p[0] for p in right_eye])
#             y_max_right = max([p[1] for p in right_eye])

#             left_eye_roi = frame[y_min_left:y_max_left, x_min_left:x_max_left]
#             right_eye_roi = frame[y_min_right:y_max_right, x_min_right:x_max_right]

#             # Preprocess eye regions for the model
#             eyes_roi = [left_eye_roi, right_eye_roi]

#             for eye_roi in eyes_roi:
#                 if eye_roi.size > 0:
#                     final_image = cv2.resize(eye_roi, (224, 224))
#                     final_image = np.expand_dims(final_image, axis=0)
#                     final_image = final_image / 255.0

#                     # Predict eye status
#                     Predictions = new_model.predict(final_image)
                    
#                     if Predictions[0][0] > 0.8:
#                         status = "Open Eyes"
#                         counter = max(0, counter - 1)
#                     else:
#                         counter += 1
#                         status = "Closed Eyes"

#     # Trigger alarm if eyes are closed for too long
#     if counter > 5:
#         status = "Sleep Alert!"
#         winsound.Beep(frequency, duration)
#         counter = 0

#     print(status,Predictions[0][0])
#     # Display status on frame
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(frame, status, (50, 50), font, 1, (0, 0, 255) if "Closed" in status else (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow('Drowsiness Detection', frame)

#     if cv2.waitKey(2) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import mediapipe as mp
import numpy as np
import winsound
import tensorflow as tf

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define beep parameters
frequency = 2500
duration = 1000

# Load your pre-trained model
new_model = tf.keras.models.load_model('my_model.keras')  # Replace with your model path

# Initialize variables
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (required by Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status = "Face Not Detected"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye landmarks (use indices from Mediapipe Face Mesh)
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            h, w, _ = frame.shape

            # Get bounding boxes for left and right eyes
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]

            # Draw eye regions on the frame
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Crop eye regions
            x_min_left = min([p[0] for p in left_eye])
            y_min_left = min([p[1] for p in left_eye])
            x_max_left = max([p[0] for p in left_eye])
            y_max_left = max([p[1] for p in left_eye])

            x_min_right = min([p[0] for p in right_eye])
            y_min_right = min([p[1] for p in right_eye])
            x_max_right = max([p[0] for p in right_eye])
            y_max_right = max([p[1] for p in right_eye])

            left_eye_roi = frame[y_min_left:y_max_left, x_min_left:x_max_left]
            right_eye_roi = frame[y_min_right:y_max_right, x_min_right:x_max_right]

            # Preprocess eye regions for the model
            eyes_roi = [left_eye_roi, right_eye_roi]

            for eye_roi in eyes_roi:
                if eye_roi.size > 0:
                    final_image = cv2.resize(eye_roi, (224, 224))
                    final_image = np.expand_dims(final_image, axis=0)
                    final_image = final_image / 255.0

                    # Predict eye status
                    Predictions = new_model.predict(final_image)
                    
                    if Predictions[0][0] >=0.8 :
                        status = "Open Eyes"
                        counter = max(0, counter - 1)
                    else:
                        counter += 1
                        status = "Closed Eyes"

    # Trigger alarm if eyes are closed for too long
    if counter > 5:
        status = "Sleep Alert!"
        winsound.Beep(frequency, duration)
        counter = 0

    print(status,Predictions[0][0])
    # Display status on frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, status, (50, 50), font, 1, (0, 0, 255) if "Closed" in status else (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

