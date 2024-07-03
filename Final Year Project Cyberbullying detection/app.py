from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)


# Function to load the model with exception handling
def load_trained_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Load the trained model
model = load_trained_model('model/my_model.keras')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500

    data = request.json
    input_data = data.get('input', None)

    if input_data is None:
        return jsonify({'error': 'Invalid input data.'}), 400

    try:
        # Preprocess the input_data if necessary
        # Assuming input_data is a list of strings, for example
        preprocessed_data = [str(input_data)]

        # Make prediction
        prediction = model.predict(preprocessed_data)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': f"Prediction error: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
