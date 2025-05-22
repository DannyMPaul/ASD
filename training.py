from spam_detec import EmailSpamDetector, load_sample_data

def main():
    print("=== Email Spam Detector Training ===")
    
    # Load the data
    print("Loading training data...")
    train_texts, train_labels, eval_texts, eval_labels = load_sample_data()
    print(f"Loaded {len(train_texts)} training examples and {len(eval_texts)} evaluation examples")
    
    # Initialize the detector
    print("\nInitializing model...")
    detector = EmailSpamDetector(model_name="distilbert-base-uncased")
    detector.load_model()
    
    # Train the model
    print("\nTraining model...")
    detector.train(
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        output_dir='./trained_spam_model',
        epochs=3,
        batch_size=8
    )
    
    print("\nTraining complete! Model saved to ./trained_spam_model")
    
    # Test with a few examples
    test_emails = [
        "URGENT: You have won a free iPhone, claim now!",
        "Meeting rescheduled to 3pm tomorrow"
    ]
    
    print("\nTesting with example emails:")
    results = detector.predict(test_emails)
    
    for result in results:
        status = "SPAM" if result["is_spam"] else "NOT SPAM"
        print(f"Email: '{result['text']}' -> {status} (Probability: {result['spam_probability']:.4f})")

if __name__ == "__main__":
    main()