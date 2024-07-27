import cv2
import numpy as np
from PIL import Image
import sys
import json
from flask import Flask, request, jsonify
import requests
from io import BytesIO

app = Flask(__name__)

def process_images(image1_url, image2_url):
    # Load the images from URLs
    response3 = requests.get(image1_url)
    response4 = requests.get(image2_url)
    image3 = Image.open(BytesIO(response3.content))
    image4 = Image.open(BytesIO(response4.content))

    # Convert images to numpy arrays
    image3_np = np.array(image3)
    image4_np = np.array(image4)

    # Convert to grayscale
    gray3 = cv2.cvtColor(image3_np, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(image4_np, cv2.COLOR_BGR2GRAY)

    # Use edge detection to highlight the shapes
    edges3 = cv2.Canny(gray3, 50, 200)
    edges4 = cv2.Canny(gray4, 50, 200)

    # Use template matching to find the location of the shape from image3 in image4
    result = cv2.matchTemplate(edges4, edges3, cv2.TM_CCOEFF_NORMED)

    # Get the coordinates of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h, w = edges3.shape

    # Calculate all four points
    bottom_right = (top_left[0] + w, top_left[1] + h)
    top_right = (top_left[0] + w, top_left[1])
    bottom_left = (top_left[0], top_left[1] + h)

    # Output the coordinates as JSON
    result_coordinates = {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_right": bottom_right,
        "bottom_left": bottom_left
    }
    
    return result_coordinates

@app.route('/match', methods=['POST'])
def match():
    # image1_url = request.form.get('image1_url')
    # image2_url = request.form.get('image2_url')
    if request.is_json:

        data = request.get_json()
        image1_url = data.get('image1_url')
        image2_url = data.get('image2_url')

        if not image1_url or not image2_url:
            return jsonify({"error": "Both 'image1_url' and 'image2_url' are required."}), 400

        try:
            result_coordinates = process_images(image1_url, image2_url)
            return jsonify(result_coordinates)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "Unsupported Media Type. Please use 'application/json'."}), 415

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
