from flask import Flask, render_template, request, flash, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
from flask import jsonify
import base64


app = Flask(__name__)

app.config['SECRET_KEY'] = 'your_secret_key_here'
# Load the pre-trained model
model = load_model('vgg19_history_10.h5')

# Mapping class indices to class names
class_names = {
    0: 'Aluminium',
    1: 'Carton',
    2: 'Glass',
    3: 'Organic',
    4: 'Other Plastics',
    5: 'Paper and Cardboard',
    6: 'Plastic',
    7: 'Textiles',
    8: 'Wood',
}
 # Function to preprocess the image before feeding it to the model
def preprocess_image(image_bytes):
    img = image.load_img(io.BytesIO(image_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class of the image
def predict_image_class(image_bytes, model):
    processed_img = preprocess_image(image_bytes)
    prediction = model.predict(processed_img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names.get(predicted_class_index, 'Unknown')
    predicted_class_probability = round(np.max(prediction) * 100, 2)  # Probability rounded to 2 decimal places
    return predicted_class, predicted_class_probability

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        try:
            # Get the image data from the request body
            image_data = request.get_json()['image']
            image_bytes = base64.b64decode(image_data)

            # Predict the class of the captured image
            predicted_class, predicted_class_probability = predict_image_class(image_bytes, model)

            return jsonify({'predicted_class': predicted_class, 'predicted_class_probability': predicted_class_probability})
        except Exception as e:
            return jsonify({'error': 'Error processing image'})

if __name__ == '__main__':
    app.run(debug=True)