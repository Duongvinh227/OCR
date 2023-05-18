import os
import json

import cv2
import numpy as np
import torch
import pytesseract
from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)

model_cccd = torch.hub.load('yolov5', 'custom', path='model_cccd.pt', source='local')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Lấy file ảnh từ request
        image = request.files['image']

        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Thay đổi kích thước ảnh
        resized_img = cv2.resize(img, (640, 640))

        # Lưu ảnh vào thư mục tạm
        cv2.imwrite('./static/uploaded_image.jpg', resized_img)

        # Chuyển hướng sang URL mới để hiển thị thông tin ảnh và JSON
        return redirect(url_for('display_image_info'))

    # Hiển thị trang HTML để tải lên ảnh
    return render_template('upload.html')


@app.route('/display_info', methods=['GET'])
def display_image_info():
    # Lấy thông tin ảnh và JSON đã tải lên
    image = cv2.imread("./static/uploaded_image.jpg")
    joined_result = info_json(image)

    # Hiển thị trang HTML với thông tin ảnh và JSON
    return render_template('image.html', result=joined_result)

def info_json(image):
    img_cccd = detection_cccd(image)
    img_id, img_name, img_birth, img_sex, img_home_town, img_residence = get_info(img_cccd)

    ocr_result = {}

    # Gọi hàm get_ocr để nhận giá trị cho từng khóa
    ocr_result["id"] = get_ocr(img_id)
    ocr_result["name"] = get_ocr(img_name)
    ocr_result["birth"] = get_ocr(img_birth)
    ocr_result["sex"] = get_ocr(img_sex)
    ocr_result["home_town"] = get_ocr(img_home_town)
    ocr_result["residence"] = get_ocr(img_residence)[7:]

    # Chuyển đổi đối tượng từ điển thành JSON
    # json_data = json.dumps(ocr_result, ensure_ascii=False)

    # Kết hợp thông tin JSON thành chuỗi
    joined_result = join_values(ocr_result)

    return joined_result


def join_values(data):
    result = {}
    for key, value in data.items():
        if isinstance(value, list):
            result[key] = " ".join(value)
        else:
            result[key] = value
    return result


def detection_cccd(image):
    detection = model_cccd(image)
    rusults = detection.pandas().xyxy[0].to_dict(orient="records")
    try:
        for result in rusults:
            # sub_dict = {key: result[key] for key in ['xmax', 'xmin', 'ymax', 'ymin']}
            # center_dict = get_center_point(sub_dict)
            confidence = round(result['confidence'], 2)
            name = result["name"]
            clas = result["class"]
            if confidence > 0.1 and name == 'top_right':
                x1 = int(result["xmin"])
                y1 = int(result["ymin"])
                x2 = int(result["xmax"])
                y2 = int(result["ymax"])
                top_right = (int((x2 + x1) / 2), int((y1 + y2) / 2))
            if confidence > 0.1 and name == 'top_left':
                x1 = int(result["xmin"])
                y1 = int(result["ymin"])
                x2 = int(result["xmax"])
                y2 = int(result["ymax"])
                top_left = (int((x2 + x1) / 2), int((y1 + y2) / 2))
            if confidence > 0.1 and name == 'bottom_right':
                x1 = int(result["xmin"])
                y1 = int(result["ymin"])
                x2 = int(result["xmax"])
                y2 = int(result["ymax"])
                bottom_right = (int((x2 + x1) / 2), int((y1 + y2) / 2))
            if confidence > 0.1 and name == 'bottom_left':
                x1 = int(result["xmin"])
                y1 = int(result["ymin"])
                x2 = int(result["xmax"])
                y2 = int(result["ymax"])
                bottom_left = (int((x2 + x1) / 2), int((y1 + y2) / 2))
        source_points = np.float32((top_left, top_right, bottom_right, bottom_left))
        img_cccd = perspective_transform(image, source_points)

        return img_cccd
    except:
        print("ảnh lỗi")

def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))
    return dst

def get_info(image):
    img_id = image[118:153, 196:385]
    img_name = image[160:190, 145:400]
    img_birth = image[184:208, 285:400]
    img_sex = image[200:230, 231:288]
    img_home_town = image[242:265, 137:487]
    img_residence = image[260:300, 141:488]
    return img_id, img_name, img_birth, img_sex,  img_home_town, img_residence

def get_ocr(image):
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    boxes = pytesseract.image_to_data(image, lang='vie')

    result = []
    for x, b in enumerate(boxes.splitlines()):
        if x != 0:
            b = b.split()
            if len(b) == 12:
                text = b[11]
                result.append(text)
    return result


if __name__ == '__main__':
    app.run(debug=True)
