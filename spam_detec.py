# Production-ready Email Spam Detection with Hugging Face
import pandas as pd
import numpy as np
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Part 1: Define the Spam Detection Model Class

class EmailSpamDetector:
    def __init__(self, model_name=None, device=None):
        """Initialize the spam detector with a pre-trained model or load a fine-tuned one."""
        # Use BERT by default or another specified model
        self.model_name = model_name or "distilbert-base-uncased"
        
        # Automatically select device (GPU if available, otherwise CPU)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
    
    def load_model(self, model_path=None):
        """Load pre-trained model or fine-tuned model from a path."""
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            print(f"Loading pre-trained model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2  # Binary classification: spam or not spam
            )
        
        self.model.to(self.device)
    
    def preprocess_data(self, texts, labels=None):
        """Preprocess text data for the model."""
        # Tokenize the texts
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=128,
            return_tensors="pt"
        )
        
        # Create a dataset
        if labels is not None:
            # For training/evaluation
            dataset = Dataset.from_dict({
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': labels
            })
        else:
            # For prediction only
            dataset = Dataset.from_dict({
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            })
        
        return dataset
    
    def compute_metrics(self, pred):
        """Compute evaluation metrics."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary'
        )
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_texts, train_labels, eval_texts=None, eval_labels=None, 
              output_dir='./spam_model', epochs=3, batch_size=16):
        """Fine-tune the model on email data."""
        # Prepare datasets
        train_dataset = self.preprocess_data(train_texts, train_labels)
        
        eval_dataset = None
        if eval_texts is not None and eval_labels is not None:
            eval_dataset = self.preprocess_data(eval_texts, eval_labels)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1" if eval_dataset else None,
            push_to_hub=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Start training
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return trainer
    
    def predict(self, texts, batch_size=16):
        """Predict whether emails are spam or not."""
        # Ensure model is loaded
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Process in batches for efficiency
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = predictions.cpu().numpy()
            
            # Extract spam probabilities
            spam_probs = predictions[:, 1]  # Assuming class 1 is spam
            for text, prob in zip(batch_texts, spam_probs):
                results.append({
                    'text': text,
                    'spam_probability': float(prob),
                    'is_spam': bool(prob > 0.5)
                })
        
        return results

# Part 2: Application Example - Using the Spam Detector

def load_sample_data():
    """Load or create sample data for demonstration."""
    try:
        # Try to load an existing dataset from Hugging Face
        dataset = load_dataset("csv", data_files={
            "train": "https://raw.githubusercontent.com/mohitgupta-omg/Spam-Classification-Dataset/master/spam.csv"
        })
        
        # Process the dataset
        train_df = pd.DataFrame(dataset["train"])
        train_df.columns = ["label", "text"]
        train_df["label"] = train_df["label"].apply(lambda x: 1 if x == "spam" else 0)
        
        # Split into train and eval
        train_df, eval_df = train_test_split(train_df, test_size=0.2, random_state=42)
        
        return (
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            eval_df["text"].tolist(),
            eval_df["label"].tolist()
        )
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using fallback sample data...")
        
        # Create a small synthetic dataset as fallback
        emails = [
            "Get rich quick! Claim your prize now!",
            "Congratulations! You've won $1,000,000",
            "FREE prescription meds at low prices",
            "Meeting scheduled for tomorrow at 10am",
            "Please review the attached report",
            "Can we discuss the project status?",
            "Buy now! Limited time offer 50% off",
            "URGENT: Your account has been suspended",
            "Hi, how are you doing today?",
            "The quarterly report is ready for review",
            "Viagra at 90% discount! Buy now!",
            "Your password will expire in 24 hours",
            "New project requirements document",
            "Double your income working from home!",
            "Team lunch scheduled for Friday"
        ]
        
        # 1 for spam, 0 for legitimate
        labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        
        # Split into train and eval
        from sklearn.model_selection import train_test_split
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            emails, labels, test_size=0.3, random_state=42
        )
        
        return train_texts, train_labels, eval_texts, eval_labels
    
# To use Own Emails dataset:
    
# def load_custom_data(data_path="your_data.csv"):
#     """Load custom email data from a CSV file."""
#     try:
#         import pandas as pd
#         from sklearn.model_selection import train_test_split
        
#         # Load the CSV file
#         print(f"Loading data from {data_path}")
#         df = pd.read_csv(data_path)
        
#         # Ensure the dataframe has the required columns
#         if 'text' not in df.columns or 'label' not in df.columns:
#             raise ValueError("CSV must contain 'text' and 'label' columns")
        
#         # Split into train and eval sets
#         train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
        
#         return (
#             train_df["text"].tolist(),
#             train_df["label"].tolist(),
#             eval_df["text"].tolist(),
#             eval_df["label"].tolist()
#         )
#     except Exception as e:
#         print(f"Error loading custom data: {e}")
#         print("Falling back to sample data...")
#         return load_sample_data()

# # Replace the call to load_sample_data() with load_custom_data() in your scripts
# # Example: train_texts, train_labels, eval_texts, eval_labels = load_custom_data("my_emails.csv")

def run_demo():
    """Run a complete demo of the spam detector."""
    # Load data
    print("Loading data...")
    train_texts, train_labels, eval_texts, eval_labels = load_sample_data()
    
    # Initialize detector
    detector = EmailSpamDetector(model_name="distilbert-base-uncased")
    detector.load_model()
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    detector.train(
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        output_dir='./spam_detector_model',
        epochs=2,  # Keep it small for demo purposes
        batch_size=8
    )
    
    # Test with some example emails
    print("\nTesting with example emails...")
    test_emails = [
        "URGENT: You have won a free iPhone, claim now!",
        "Meeting rescheduled to 3pm tomorrow",
        "Double your money in just one week guaranteed!",
        "Please review the quarterly financial report",
        "Your Amazon package will be delivered today"
    ]
    
    results = detector.predict(test_emails)
    
    # Display results
    print("\nPrediction Results:")
    for result in results:
        status = "SPAM" if result["is_spam"] else "NOT SPAM"
        print(f"Email: '{result['text'][:50]}...' -> {status} (Probability: {result['spam_probability']:.4f})")

# Part 3: Deploying the Model as a REST API

def create_api():
    """Create a Flask API to serve the spam detection model."""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # Initialize the model
    detector = EmailSpamDetector()
    model_path = './spam_detector_model'  # Path to your trained model
    
    # Check if we have a fine-tuned model, otherwise use pre-trained
    if os.path.exists(model_path):
        detector.load_model(model_path)
    else:
        detector.load_model()
        print("Warning: Using pre-trained model without fine-tuning")
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy"})
    
    @app.route('/predict', methods=['POST'])
    def predict_spam():
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
    
    # Code to start the server
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)
    
    return app

# Main execution - uncomment the desired function
if __name__ == "__main__":
    # For training and testing
    run_demo()
    
    # To deploy the API (uncomment to use)
    # app = create_api()
    # app.run(host='0.0.0.0', port=5000)