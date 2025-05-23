from spam_detec import EmailSpamDetector
import os

def load_sample_data():
    """Load sample data for demonstration."""
    print("Using sample data for training...")
    
    # Extended sample dataset with more examples
    emails = [
        # Spam emails
        "Get rich quick! Claim your prize now!",
        "Congratulations! You've won $1,000,000 dollars!",
        "FREE prescription meds at low prices - no prescription needed",
        "Buy now! Limited time offer 50% off everything",
        "URGENT: Your account has been suspended - click here immediately",
        "Viagra at 90% discount! Buy now and save big!",
        "Double your income working from home! No experience required",
        "You have been selected for a special offer - act now!",
        "WINNER! You have won a free iPhone - claim within 24 hours",
        "Make money fast! Earn $5000 per week from home",
        "Nigerian Prince needs your help - millions waiting for you",
        "Free casino bonus - claim your $500 now",
        "Weight loss miracle - lose 30 pounds in 30 days",
        "Hot singles in your area want to meet you",
        "Refinance your mortgage - lowest rates guaranteed",
        
        # Legitimate emails
        "Meeting scheduled for tomorrow at 10am in conference room A",
        "Please review the attached quarterly report by Friday",
        "Can we discuss the project status in our 1:1 meeting?",
        "The quarterly financial results are now available",
        "Team lunch scheduled for Friday at 12:30pm",
        "Your password will expire in 7 days - please update it",
        "New project requirements document has been shared",
        "Hi, how are you doing? Hope you're having a great week",
        "The server maintenance is scheduled for this weekend",
        "Please submit your timesheet by end of day",
        "Meeting minutes from yesterday's call are attached",
        "Your order has been shipped and will arrive in 2-3 days",
        "Thank you for your presentation yesterday",
        "The new employee handbook is now available online",
        "System update completed successfully last night"
    ]
    
    # Labels: 1 for spam, 0 for legitimate
    labels = [
        # Spam labels (first 15 emails)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # Legitimate labels (next 15 emails)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    
    # Split into train and eval using sklearn
    from sklearn.model_selection import train_test_split
    
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        emails, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    return train_texts, train_labels, eval_texts, eval_labels

def main():
    print("=== Email Spam Detector Training ===")
    
    try:
        # Load the data
        print("Loading training data...")
        train_texts, train_labels, eval_texts, eval_labels = load_sample_data()
        print(f"Loaded {len(train_texts)} training examples and {len(eval_texts)} evaluation examples")
        
        # Initialize the detector
        print("\nInitializing model...")
        detector = EmailSpamDetector(model_name="distilbert-base-uncased")
        detector.load_model()

    except:
        print("Error loading data or initializing model. Please check your setup.")
        return
    
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