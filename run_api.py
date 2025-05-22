from flask import Flask, request, jsonify
import os
from spam_detector import EmailSpamDetector

# Initialize Flask app
app = Flask(__name__)

# Global detector instance
detector = None

def initialize_model():
    """Initialize the spam detector model."""
    global detector
    
    print("Initializing spam detector model...")
    detector = EmailSpamDetector()
    
    # Check if we have a fine-tuned model
    model_path = './trained_spam_model'
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        detector.load_model(model_path)
    else:
        print("No fine-tuned model found. Loading pre-trained model...")
        detector.load_model()
    
    print("Model initialization complete!")

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the service is running."""
    return jsonify({"status": "healthy", "model_loaded": detector is not None})

@app.route('/predict', methods=['POST'])
def predict_spam():
    """Endpoint to predict if emails are spam."""
    # Check if model is loaded
    global detector
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Get data from request
    data = request.json
    
    if not data or 'emails' not in data:
        return jsonify({"error": "Missing 'emails' field in request"}), 400
    
    emails = data['emails']
    
    # Make predictions
    try:
        results = detector.predict(emails)
        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Endpoint with example predictions for testing."""
    global detector
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    test_emails = [
        "URGENT: You have won a free iPhone, claim now!",
        "Meeting rescheduled to 3pm tomorrow"
    ]
    
    results = detector.predict(test_emails)
    return jsonify({"test_predictions": results})

if __name__ == '__main__':
    # Initialize the model before starting the server
    initialize_model()
    
    # Run the Flask app
    print("Starting API server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)