import serial
import time
import mediapipe as mp
import numpy as np
import cv2
import winsound
import tensorflow as tf

def camera():
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
                        
                        if Predictions[0][0] > 0.8:
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

# Set the correct serial port for your system
arduino_port = "COM3"  # Update this to your port (e.g., COM3, COM4)
baud_rate = 115200  # Same as in the Arduino code

# Create a serial connection to the Arduino
arduino = serial.Serial(arduino_port, baud_rate, timeout=1)

time.sleep(3)  # Wait for the Arduino to initialize

hardcoded_sleep = [46, 59, 44, 50, 46, 52, 58, 51, 56, 55, 60, 49]



while True:
    print("If you want to get the real time data for 3 cycles Enter 0")
    print("If you want to enter the Hardcoded sleep data enter 1")
    print("Enter 2 to Exit")
    val=int(input("Enter your response above. "))
    ctr=3
    if(val==0):
        print("Enter Input Above.")
        while(ctr>0):
            try:
                bpm_readings = []  # List to store BPM readings for 10 seconds
                start_time = time.time()  # Record the start time
                while time.time() - start_time < 10: #collecting for 10 Seconds
                    if arduino.in_waiting > 0:
                        bpm_data = (
                            arduino.readline().decode("utf-8").strip()
                        )  # Read a line from the Arduino
                        print(f"Received data: {bpm_data}") 
                        try:
                            bpm_value = int(bpm_data)
                            bpm_readings.append(bpm_value)
                        except ValueError:
                            print(f"Invalid data: {bpm_data}") 
                            continue
                if bpm_readings:
                    average_bpm = sum(bpm_readings) / len(bpm_readings) #Avg BPM
                else:
                    average_bpm = 0
                print(bpm_readings)
                print()
                # Categorize the BPM based on the range
                if average_bpm >= 40 and average_bpm <= 65:
                    activity_level = "Resting (Drowsy/Sleeping)"
                    print(f"Average BPM for the last 10 seconds: {average_bpm:.2f}")
                    print(f"Activity Level: {activity_level}\n")
                    # give function call here for camera code call
                    # give function call here for camera code call
                    # give function call here for camera code call
                    # give function call here for camera code call
                    # give function call here for camera code call
                    # give function call here for camera code call
                    # give function call here too for sleep
                elif average_bpm >= 66 and average_bpm <= 130:
                    activity_level = f"You are awake! "
                elif average_bpm >= 131:
                    activity_level = "Intense Exercise and Awake "

                # Display the results
                print(f"Average BPM for the last 10 seconds: {average_bpm:.2f}")
                print(f"Activity Level: {activity_level}\n")
                ctr-=1
                time.sleep(0.1)  # Small delay to prevent overwhelming the serial connection
            

            except KeyboardInterrupt:
                print("Exiting...")
                break

    elif(val ==1):
        average_bpm = sum(hardcoded_sleep) / len(hardcoded_sleep)
        activity_level = "Resting (Drowsy/Sleeping)"
        print(f"Average BPM for the last 10 seconds: {average_bpm:.2f}")
        print(f"Activity Level: {activity_level}\n")
        print("Now Initializing the camera detection: ")
        ##camera code call here

    elif(val==2):
        quit()