import cv2
import requests
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os;
port = int(os.environ.get("PORT", 443))

app = Flask(__name__)

# Đường dẫn đến ảnh từ Cloudinary
CLOUDINARY_URL = "https://res.cloudinary.com/dphfcojlc/image/upload/{image_name}.jpg"

# Tải ảnh từ Cloudinary
def load_image_from_cloud(image_name):
    url = CLOUDINARY_URL.format(image_name=image_name)
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return np.array(image)

# Hàm phát hiện khuôn mặt sử dụng OpenCV
def detect_faces(image_np):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Kiểm tra khuôn mặt có xuất hiện trong danh sách ảnh cloud không
def match_face_with_cloud_images(image_np, cloud_image_names):
    user_faces = detect_faces(image_np)
    if len(user_faces) == 0:
        return False  # Không tìm thấy khuôn mặt

    for cloud_image_name in cloud_image_names:
        cloud_image = load_image_from_cloud(cloud_image_name)
        cloud_faces = detect_faces(cloud_image)

        if len(cloud_faces) > 0:
            return True  # Nếu có khuôn mặt trong ảnh cloud, coi như hợp lệ

    return False

@app.route('/verify', methods=['POST'])
def verify_face():
    data = request.json
    image_data = base64.b64decode(data['image'])
    image = Image.open(BytesIO(image_data))
    image_np = np.array(image)

    # Danh sách tên ảnh người quen trong cloud
    cloud_image_names = data['image_names']  # Danh sách tên ảnh

    if match_face_with_cloud_images(image_np, cloud_image_names):
        return jsonify({"status": "success", "message": "Face verified"})

    return jsonify({"status": "failure", "message": "Face not recognized"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
