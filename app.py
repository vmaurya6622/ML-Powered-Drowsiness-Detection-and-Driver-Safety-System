from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound
import threading

# Initialize Flask app
app = Flask(__name__)

# Load the fine-tuned model
model = load_model("my_model.keras")

# Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Alarm settings
alarm_sound_path = "alarm.wav"
counter = 0

def detect_drowsiness(frame):
    global counter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eyes_roi = roi_color[ey:ey + eh, ex:ex + ew]
            try:
                final_image = cv2.resize(eyes_roi, (224, 224))
                final_image = np.expand_dims(final_image, axis=0)
                final_image = final_image / 255.0

                Predictions = model.predict(final_image)
                font = cv2.FONT_HERSHEY_SIMPLEX
                print(Predictions)
                if Predictions < 0.4:
                    counter = 0
                    
                    cv2.putText(frame, "Open Eyes", (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    counter += 1
                    cv2.putText(frame, "Closed Eyes", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    if counter > 5:
                        cv2.putText(frame, "Sleep Alert!", (50, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        threading.Thread(target=playsound, args=(alarm_sound_path,)).start()
                        counter = 0
            except Exception as e:
                pass
    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Drowsiness detection
        frame = detect_drowsiness(frame)

        # Encode frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in byte format for streaming
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)