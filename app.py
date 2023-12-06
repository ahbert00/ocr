import pytesseract
import shutil
import os
import random
import cv2
import numpy as np
import base64
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# initialize the trOCR
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained(
    'microsoft/trocr-base-handwritten')
decoded_image = None


def ocr_text(image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]

    return generated_text


def get_text_coordinations(image):
    lines = pytesseract.image_to_data(image)
    print(lines)

    # convert the string into a 2d array    
    temp = lines.split("\n")
    final_list = []
    for i in temp:
        final_list.append(i.split("\t"))

    # remove the first and last item which are empty list
    final_list.pop(0)
    final_list.pop(-1)

    categorized_list = []
    row_list = []
    # categorize item from same row into a list
    for item in final_list:
        if (item[5] != '0'):
            row_list.append(item)
        else:
            categorized_list.append(row_list)
            row_list = []

    print(categorized_list)

    # remove list item that are empty
    filtered_list = list(filter(None, categorized_list))

    # convert numeric string to integer
    converted_list = [[[int(value) if value.isdigit() else value for value in inner_list]
                       for inner_list in outer_list] for outer_list in filtered_list]

    # list to store all the coordinate of the text rows
    location = []

    # extract top left and bottom right corner of the scan result from the result
    for item in converted_list:
        min_based_on_6 = sorted(item, key=lambda x: x[6], reverse=False)
        min_based_on_7 = sorted(item, key=lambda x: x[7], reverse=False)
        max_based_on_6 = sorted(item, key=lambda x: x[6], reverse=True)
        max_based_on_7 = sorted(item, key=lambda x: x[7], reverse=True)

        # add padding for better crop
        margin = 10
        # prevent negative value
        tlx = 0 if ((min_based_on_6[0][6] - margin)
                    < 0) else (min_based_on_6[0][6] - margin)
        tly = 0 if ((min_based_on_7[0][7] - margin)
                    < 0) else (min_based_on_7[0][7] - margin)
        brx = max_based_on_6[0][6] + max_based_on_6[0][8] + margin
        bry = max_based_on_7[0][7] + max_based_on_7[0][9] + margin

        location.append([tlx, tly, brx, bry])

    print(location)
    return location


@app.route("/")
def index():
    return "Hello"


def stringToImage(base64_string):
    im_bytes = base64.b64decode(base64_string)
    # im_arr is one-dim Numpy array
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    return cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)


@app.route("/ocr", methods=['GET', 'POST'])
def scanner():
    global decoded_image

    if request.method == 'POST':
        base64_string = request.form['base64']

        # image_path = "C:/Users/Asus/Downloads/level2.jpg"

        # decoded_image = cv2.imread(image_path)

        # image_bytes = bytes(base64_string, encoding="ascii")
        # Convert the bytes to a NumPy array
        # image_np = np.frombuffer(image_bytes, dtype=np.uint8)

        # Decode the NumPy array to an OpenCV image
        # decoded_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        decoded_image = stringToImage(base64_string)

        # print(decoded_image)
        location = get_text_coordinations(decoded_image)

        ocr_text_list = []

        for r in location:

            start_x = r[0]
            start_y = r[1]
            height = r[3]-r[1]
            width = r[2]-r[0]

            crop = decoded_image[start_y:start_y +
                                 height, start_x: start_x + width]

            ocr_text_list.append(ocr_text(crop))

        # return base64
        json = {
            "result": ocr_text_list,
        }
        return json

    # return "Hello"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)