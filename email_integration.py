import imaplib
import email
import time
import json
from datetime import datetime
from spam_detec import EmailSpamDetector
import os

class EmailSpamChecker:
    def __init__(self, model_path=None):
        """Initialize the email spam checker."""
        self.detector = EmailSpamDetector()
        
        # Load trained model if available
        if model_path and os.path.exists(model_path):
            self.detector.load_model(model_path)
            print(f"Loaded trained model from {model_path}")
        else:
            self.detector.load_model()
            print("Using pre-trained model")
        
        self.imap_server = None
        self.email_config = {}
    
    def setup_email_connection(self, email_address, password, imap_server, imap_port=993):
        """Set up connection to email server."""
        self.email_config = {
            'email': email_address,
            'password': password,
            'imap_server': imap_server,
            'imap_port': imap_port
        }
        
        try:
            # Connect to the server
            self.imap_server = imaplib.IMAP4_SSL(imap_server, imap_port)
            self.imap_server.login(email_address, password)
            print(f"Successfully connected to {email_address}")
            return True
        except Exception as e:
            print(f"Failed to connect to email: {str(e)}")
            return False
    
    def extract_email_content(self, email_message):
        """Extract text content from email message."""
        content = ""
        subject = email_message.get('Subject', '')
        
        # Add subject to content
        content += f"Subject: {subject}\n\n"
        
        # Extract body
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode('utf-8')
                        content += body
                        break
                    except:
                        continue
        else:
            try:
                body = email_message.get_payload(decode=True).decode('utf-8')
                content += body
            except:
                content += "Could not decode email body"
        
        return content.strip()
    
    def check_recent_emails(self, folder='INBOX', days=1, max_emails=50):
        """Check recent emails for spam."""
        if not self.imap_server:
            print("No email connection established")
            return []
        
        try:
            # Select the folder
            self.imap_server.select(folder)
            
            # Search for recent emails
            search_criteria = f'(SINCE "{(datetime.now().strftime("%d-%b-%Y"))}")'
            result, email_ids = self.imap_server.search(None, search_criteria)
            
            if result != 'OK':
                print("No emails found")
                return []
            
            email_ids = email_ids[0].split()
            if not email_ids:
                print("No recent emails found")
                return []
            
            # Limit number of emails to check
            email_ids = email_ids[-max_emails:] if len(email_ids) > max_emails else email_ids
            
            results = []
            emails_to_check = []
            email_details = []
            
            print(f"Checking {len(email_ids)} recent emails...")
            
            # Fetch emails
            for email_id in email_ids:
                try:
                    result, email_data = self.imap_server.fetch(email_id, '(RFC822)')
                    if result == 'OK':
                        email_message = email.message_from_bytes(email_data[0][1])
                        
                        # Extract email details
                        sender = email_message.get('From', 'Unknown')
                        subject = email_message.get('Subject', 'No Subject')
                        date = email_message.get('Date', 'Unknown')
                        content = self.extract_email_content(email_message)
                        
                        emails_to_check.append(content)
                        email_details.append({
                            'id': email_id.decode(),
                            'sender': sender,
                            'subject': subject,
                            'date': date,
                            'content_preview': content[:200] + "..." if len(content) > 200 else content
                        })
                
                except Exception as e:
                    print(f"Error processing email {email_id}: {str(e)}")
                    continue
            
            # Run spam detection on all emails at once
            if emails_to_check:
                spam_predictions = self.detector.predict(emails_to_check)
                
                # Combine results
                for i, (prediction, details) in enumerate(zip(spam_predictions, email_details)):
                    results.append({
                        'email_details': details,
                        'spam_prediction': prediction
                    })
            
            return results
            
        except Exception as e:
            print(f"Error checking emails: {str(e)}")
            return []
    
    def scan_and_report(self, folder='INBOX', days=1, max_emails=50):
        """Scan emails and generate a report."""
        results = self.check_recent_emails(folder, days, max_emails)
        
        if not results:
            print("No emails to analyze")
            return
        
        spam_count = sum(1 for r in results if r['spam_prediction']['is_spam'])
        total_count = len(results)
        
        print(f"\n=== SPAM DETECTION REPORT ===")
        print(f"Total emails checked: {total_count}")
        print(f"Spam detected: {spam_count}")
        print(f"Legitimate emails: {total_count - spam_count}")
        print("="*50)
        
        # Show spam emails
        if spam_count > 0:
            print("\n🚨 POTENTIAL SPAM EMAILS:")
            for result in results:
                if result['spam_prediction']['is_spam']:
                    details = result['email_details']
                    prediction = result['spam_prediction']
                    
                    print(f"\nFrom: {details['sender']}")
                    print(f"Subject: {details['subject']}")
                    print(f"Date: {details['date']}")
                    print(f"Spam Confidence: {prediction['spam_probability']:.2%}")
                    print(f"Preview: {details['content_preview']}")
                    print("-" * 40)
        
        # Show legitimate emails with high confidence
        print(f"\n✅ LEGITIMATE EMAILS:")
        legit_count = 0
        for result in results:
            if not result['spam_prediction']['is_spam'] and legit_count < 5:  # Show first 5
                details = result['email_details']
                prediction = result['spam_prediction']
                
                print(f"From: {details['sender']}")
                print(f"Subject: {details['subject']}")
                print(f"Confidence: {(1-prediction['spam_probability']):.2%}")
                print("-" * 30)
                legit_count += 1
        
        if total_count - spam_count > 5:
            print(f"... and {total_count - spam_count - 5} more legitimate emails")
    
    def continuous_monitoring(self, check_interval=300):  # 5 minutes
        """Continuously monitor for new emails."""
        print(f"Starting continuous monitoring (checking every {check_interval//60} minutes)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for new emails...")
                self.scan_and_report(max_emails=10)  # Check last 10 emails
                
                print(f"Sleeping for {check_interval//60} minutes...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error during monitoring: {str(e)}")
    
    def close_connection(self):
        """Close email connection."""
        if self.imap_server:
            try:
                self.imap_server.close()
                self.imap_server.logout()
                print("Email connection closed")
            except:
                pass

# Example usage and configuration for different email providers
def get_email_config():
    """Get email configuration for popular providers."""
    configs = {
        'gmail': {
            'imap_server': 'imap.gmail.com',
            'imap_port': 993,
            'note': 'Use App Password instead of regular password for Gmail'
        },
        'outlook': {
            'imap_server': 'outlook.office365.com',
            'imap_port': 993,
            'note': 'Regular password should work for Outlook'
        },
        'yahoo': {
            'imap_server': 'imap.mail.yahoo.com',
            'imap_port': 993,
            'note': 'Use App Password for Yahoo Mail'
        },
        'apple': {
            'imap_server': 'imap.mail.me.com',
            'imap_port': 993,
            'note': 'Use App Password for iCloud Mail'
        }
    }
    
    print("Supported email providers:")
    for provider, config in configs.items():
        print(f"{provider}: {config['imap_server']} - {config['note']}")
    
    return configs

def main():
    """Main function to run email spam checking."""
    print("=== Personal Email Spam Checker ===")
    
    # Show supported providers
    configs = get_email_config()
    
    # Get user input
    email_address = input("\nEnter your email address: ")
    password = input("Enter your password (or app password): ")
    
    # Determine provider
    provider = None
    for p in configs.keys():
        if p in email_address.lower():
            provider = p
            break
    
    if not provider:
        print("Provider not automatically detected. Using Gmail settings as default.")
        provider = 'gmail'
    
    config = configs[provider]
    print(f"Using {provider} configuration: {config['imap_server']}")
    
    # Initialize spam checker
    model_path = './trained_spam_model' if os.path.exists('./trained_spam_model') else None
    spam_checker = EmailSpamChecker(model_path)
    
    # Connect to email
    if spam_checker.setup_email_connection(
        email_address, 
        password, 
        config['imap_server'], 
        config['imap_port']
    ):
        print("\nChoose an option:")
        print("1. Check recent emails once")
        print("2. Continuous monitoring")
        
        choice = input("Enter choice (1 or 2): ")
        
        if choice == '1':
            # One-time check
            days = input("Check emails from how many days ago? (default: 1): ")
            days = int(days) if days.isdigit() else 1
            
            max_emails = input("Maximum emails to check? (default: 50): ")
            max_emails = int(max_emails) if max_emails.isdigit() else 50
            
            spam_checker.scan_and_report(days=days, max_emails=max_emails)
            
        elif choice == '2':
            # Continuous monitoring
            interval = input("Check interval in minutes? (default: 5): ")
            interval = int(interval) * 60 if interval.isdigit() else 300
            
            spam_checker.continuous_monitoring(interval)
        
        spam_checker.close_connection()
    else:
        print("Failed to connect to email server. Please check your credentials.")

if __name__ == "__main__":
    main()