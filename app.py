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

model1 = load_model('vgg19_history_10.h5')
model2 = load_model('model_with_history.h5')

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

def preprocess_image(image_bytes):
    img = image.load_img(io.BytesIO(image_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image_class(image_bytes, model1, model2, threshold=5):
    processed_img = preprocess_image(image_bytes)
    prediction1 = model1.predict(processed_img)
    prediction2 = model2.predict(processed_img)

    combined_prediction = (prediction1 + prediction2) / 2

    predicted_class_indices = combined_prediction.argsort()[0][-2:]
    predicted_classes = [class_names.get(index, 'Unknown') for index in predicted_class_indices]
    predicted_probabilities = [round(combined_prediction[0, index] * 100, 2) for index in predicted_class_indices]


    if abs(predicted_probabilities[0] - predicted_probabilities[1]) <= threshold:
        return predicted_classes, predicted_probabilities
    else:
        max_probability_index = np.argmax(combined_prediction)
        max_probability_class = class_names.get(max_probability_index, 'Unknown')
        return [max_probability_class], [round(np.max(combined_prediction) * 100, 2)]



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        try:
            image_data = request.get_json()['image']
            image_bytes = base64.b64decode(image_data)
            predicted_class, predicted_class_probability = predict_image_class(image_bytes, model1, model2)
            return jsonify({'predicted_class': predicted_class, 'predicted_class_probability': predicted_class_probability})
        except Exception as e:
            return jsonify({'error': 'Error processing image'})

if __name__ == '__main__':
    app.run(debug=True)
