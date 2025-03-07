import cv2
import cloudinary
import cloudinary.uploader
import cloudinary.api
from flask import Flask, request
import numpy as np
from io import BytesIO



# Cấu hình Cloudinary
cloudinary.config(
    cloud_name='dphfcojlc',
    api_key='133562825661752',
    api_secret='f1UBV7B7jR03USGeqgeuRuXKdnI'
)

# Tải cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Hàm xử lý ảnh và lưu vào Cloudinary
def process_and_upload_face(frame):
    # Chuyển ảnh sang màu xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Cắt khuôn mặt
        face = frame[y:y+h, x:x+w]
        
         # Chuyển ảnh khuôn mặt thành định dạng mà Cloudinary có thể nhận
        _, buffer = cv2.imencode('.jpg', face)
        face_data = BytesIO(buffer)
         # Upload ảnh lên Cloudinary
        upload_result = cloudinary.uploader.upload(
            face_data,
            resource_type="image",  # Tài nguyên là ảnh
            folder="load/image",  # Thư mục 'load/image'
        )
        print(f"Ảnh đã được upload lên Cloudinary: {upload_result['secure_url']}")
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị ảnh với các khuôn mặt được phát hiện
    cv2.imshow('Capture Face Samples', frame)

    return frame


app = Flask(__name__)

@app.route('/capture_face', methods=['POST'])
def capture_face():
    # Lấy ảnh từ client
    file = request.files['image']
    # Đọc ảnh từ file
    frame = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Xử lý và upload ảnh khuôn mặt
    process_and_upload_face(frame)
    
    return 'Processed and uploaded face images.'

if __name__ == "__main__":
    app.run(debug=True,port= 8080)
