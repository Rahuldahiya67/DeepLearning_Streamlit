# tumor_detection_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

class TumorDetectionModel:
    def __init__(self):
        # Load a pre-trained MobileNetV2 model for image classification
        self.model = MobileNetV2(weights='imagenet')

    def preprocess_image(self, img_path):
        # Load and preprocess the image for model input
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        return img_array

    def detect_tumor(self, image_file):
        # Ensure the file is an image
        if not image_file.type.startswith('image'):
            return "Invalid file type. Please upload an image."

        # Preprocess the image
        img_array = self.preprocess_image(image_file)

        # Get model predictions
        predictions = self.model.predict(img_array)

        # Decode predictions
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Return the top prediction
        return decoded_predictions[0][1]

# Example of using the model
if __name__ == "__main__":
    tumor_model = TumorDetectionModel()
    img_path = "path_to_your_image.jpg"  # Replace with the path to your image
    result = tumor_model.detect_tumor(img_path)
    print(f"Tumor detection result: {result}")
