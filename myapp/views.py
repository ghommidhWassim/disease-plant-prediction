import os
import json  # Add this import
import numpy as np
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from tensorflow.keras.models import load_model
import logging
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

# Load the pre-trained model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join('/home/gass/Desktop/plant-disease-detection/plant_disease_detection/myapp/plant_disease_model.h5')
model = load_model(model_path)

# Load class indices
class_indices_path = os.path.join('/home/gass/Desktop/plant-disease-detection/plant_disease_detection/myapp/class_indices.json')
with open(class_indices_path) as f:
    class_indices = json.load(f)  # Now this will work

# Function to load and preprocess the image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to encode image as base64
def encode_image_as_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Django view for handling image upload and prediction
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        try:
            # Get the uploaded image
            uploaded_file = request.FILES['image']
            image = Image.open(uploaded_file)

            # Encode the image as base64
            image_base64 = encode_image_as_base64(image)

            # Preprocess the image and get the prediction
            preprocessed_img = load_and_preprocess_image(uploaded_file)
            predictions = model.predict(preprocessed_img)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_indices[str(predicted_class_index)]

            # Return JSON response with base64 image and prediction
            return JsonResponse({
                'image_base64': image_base64,
                'prediction': predicted_class_name,
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return render(request, 'myapp/upload.html')