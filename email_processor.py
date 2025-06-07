from flask import Flask, request, jsonify
import os
import json
from datetime import datetime
from spam_detec import EmailSpamDetector

app = Flask(__name__)

# Initialize spam detector
detector = EmailSpamDetector()
model_path = './trained_spam_model' if os.path.exists('./trained_spam_model') else None
detector.load_model(model_path)

# Store results for review
RESULTS_FILE = 'spam_results.json'

def load_results():
    """Load previous results."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_result(result):
    """Save spam detection result."""
    results = load_results()
    results.append({
        'timestamp': datetime.now().isoformat(),
        'result': result
    })
    
    # Keep only last 100 results
    results = results[-100:]
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

@app.route('/check-email', methods=['POST'])
def check_email():
    """Endpoint to check a single email."""
    data = request.json
    
    if not data or 'email_content' not in data:
        return jsonify({"error": "Missing email_content"}), 400
    
    email_content = data['email_content']
    sender = data.get('sender', 'Unknown')
    subject = data.get('subject', 'No Subject')
    
    # Combine subject and content for analysis
    full_content = f"Subject: {subject}\nFrom: {sender}\n\n{email_content}"
    
    # Get prediction
    result = detector.predict([full_content])[0]
    
    # Add metadata
    result['sender'] = sender
    result['subject'] = subject
    result['timestamp'] = datetime.now().isoformat()
    
    # Save result
    save_result(result)
    
    return jsonify(result)

@app.route('/results', methods=['GET'])
def get_results():
    """Get recent spam detection results."""
    results = load_results()
    
    # Summary statistics
    total = len(results)
    spam_count = sum(1 for r in results if r['result']['is_spam'])
    
    return jsonify({
        'total_checked': total,
        'spam_detected': spam_count,
        'recent_results': results[-20:]  # Last 20 results
    })

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Simple web dashboard to view results."""
    results = load_results()
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Email Spam Detection Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .spam { background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; }
            .legitimate { background-color: #e8f5e8; border-left: 4px solid #4caf50; padding: 10px; margin: 10px 0; }
            .stats { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>Email Spam Detection Dashboard</h1>
    """
    
    if results:
        total = len(results)
        spam_count = sum(1 for r in results if r['result']['is_spam'])
        
        html += f"""
        <div class="stats">
            <h3>Statistics</h3>
            <p>Total emails checked: {total}</p>
            <p>Spam detected: {spam_count}</p>
            <p>Legitimate emails: {total - spam_count}</p>
            <p>Spam rate: {(spam_count/total*100):.1f}%</p>
        </div>
        
        <h3>Recent Results</h3>
        """
        
        for entry in results[-20:]:  # Show last 20
            result = entry['result']
            timestamp = entry['timestamp']
            
            css_class = 'spam' if result['is_spam'] else 'legitimate'
            status = 'SPAM' if result['is_spam'] else 'LEGITIMATE'
            
            html += f"""
            <div class="{css_class}">
                <strong>{status}</strong> - {timestamp}<br>
                <strong>From:</strong> {result.get('sender', 'Unknown')}<br>
                <strong>Subject:</strong> {result.get('subject', 'No Subject')}<br>
                <strong>Confidence:</strong> {result['spam_probability']:.2%}<br>
                <em>Preview:</em> {result['text'][:150]}...
            </div>
            """
    else:
        html += "<p>No emails checked yet.</p>"
    
    html += """
        <script>
            // Auto-refresh every 30 seconds
            setTimeout(function(){ location.reload(); }, 30000);
        </script>
    </body>
    </html>
    """
    
    return html

if __name__ == '__main__':
    print("Email Processor API started!")
    print("Endpoints:")
    print("- POST /check-email: Check a single email")
    print("- GET /results: Get detection results as JSON")
    print("- GET /dashboard: View web dashboard")
    print("\nDashboard available at: http://localhost:5001/dashboard")
    
    app.run(host='0.0.0.0', port=5001, debug=False)