#!/usr/bin/env python3
"""
Master Startup Script for Email Spam Detection System
This script orchestrates the entire spam detection pipeline
"""

import os
import sys
import subprocess
import time
import threading
import signal
from pathlib import Path

class SpamDetectionSystem:
    def __init__(self):
        self.processes = {}
        self.system_ready = False
        
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            'transformers', 'torch', 'flask', 'sklearn', 
            'pandas', 'numpy', 'datasets', 'accelerate'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("Run: pip install -r requirements.txt")
            return False
        
        print("✅ All dependencies installed")
        return True
    
    def check_model_status(self):
        """Check if a trained model exists."""
        model_path = Path('./trained_spam_model')
        if model_path.exists() and any(model_path.iterdir()):
            print("✅ Trained model found")
            return True
        else:
            print("⚠️  No trained model found")
            return False
    
    def train_model(self):
        """Train the spam detection model."""
        print("\n🎯 Starting model training...")
        print("This may take several minutes...")
        
        try:
            result = subprocess.run([sys.executable, 'run_training.py'], 
                                  capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print("✅ Model training completed successfully")
                return True
            else:
                print(f"❌ Training failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Training timed out (1 hour limit)")
            return False
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            return False
    
    def start_api_server(self):
        """Start the basic API server."""
        print("🚀 Starting API server...")
        
        try:
            process = subprocess.Popen([sys.executable, 'run_api.py'])
            self.processes['api'] = process
            
            # Wait a bit for server to start
            time.sleep(3)
            
            # Test if server is running
            import requests
            try:
                response = requests.get('http://localhost:5000/health', timeout=5)
                if response.status_code == 200:
                    print("✅ API server started on http://localhost:5000")
                    return True
            except:
                pass
            
            print("⚠️  API server may have issues - check manually")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start API server: {str(e)}")
            return False
    
    def start_email_processor(self):
        """Start the email processor with dashboard."""
        print("📧 Starting email processor with dashboard...")
        
        try:
            process = subprocess.Popen([sys.executable, 'email_processor.py'])
            self.processes['processor'] = process
            
            # Wait a bit for server to start
            time.sleep(3)
            
            print("✅ Email processor started with dashboard on http://localhost:5001")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start email processor: {str(e)}")
            return False
    
    def show_system_status(self):
        """Display the current system status."""
        print("\n" + "="*60)
        print("📊 SPAM DETECTION SYSTEM STATUS")
        print("="*60)
        
        # Check processes
        running_services = []
        for service, process in self.processes.items():
            if process.poll() is None:  # Process is running
                running_services.append(service)
        
        print(f"🔄 Running services: {', '.join(running_services) if running_services else 'None'}")
        
        # Show available endpoints
        if 'api' in running_services:
            print("\n🌐 API Endpoints:")
            print("   • Health Check: http://localhost:5000/health")
            print("   • Spam Detection: POST http://localhost:5000/predict")
            print("   • Test Endpoint: http://localhost:5000/test")
        
        if 'processor' in running_services:
            print("\n📊 Email Processing:")
            print("   • Dashboard: http://localhost:5001/dashboard")
            print("   • Check Email: POST http://localhost:5001/check-email")
            print("   • Results API: http://localhost:5001/results")
        
        print("\n📧 Available Tools:")
        print("   • Direct email checking: python email_integration.py")
        print("   • Manual training: python run_training.py")
        
        print("\n" + "="*60)
    
    def run_interactive_mode(self):
        """Run interactive mode for user choices."""
        while True:
            print("\n🎮 INTERACTIVE MODE")
            print("Choose an option:")
            print("1. Check system status")
            print("2. Train/retrain model")
            print("3. Test API with sample emails")
            print("4. Check your personal emails")
            print("5. View dashboard")
            print("6. Stop all services")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                self.show_system_status()
                
            elif choice == '2':
                self.train_model()
                
            elif choice == '3':
                self.test_api()
                
            elif choice == '4':
                self.run_email_checker()
                
            elif choice == '5':
                import webbrowser
                webbrowser.open('http://localhost:5001/dashboard')
                print("📊 Dashboard opened in your browser")
                
            elif choice == '6':
                self.stop_all_services()
                print("🛑 All services stopped")
                
            elif choice == '7':
                self.stop_all_services()
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
    
    def test_api(self):
        """Test the API with sample emails."""
        print("\n🧪 Testing API with sample emails...")
        
        try:
            import requests
            
            test_emails = [
                "URGENT: You have won a lottery! Claim your $1,000,000 prize now!",
                "Meeting rescheduled to tomorrow at 3pm in conference room B",
                "Get rich quick! Make money from home with this amazing opportunity!",
                "Please review the quarterly report attached to this email"
            ]
            
            response = requests.post(
                'http://localhost:5000/predict',
                json={'emails': test_emails},
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()['predictions']
                
                print("\n📊 Test Results:")
                for result in results:
                    status = "🚨 SPAM" if result['is_spam'] else "✅ SAFE"
                    confidence = result['spam_probability'] * 100
                    email_preview = result['text'][:50] + "..."
                    
                    print(f"{status} ({confidence:.1f}%) - {email_preview}")
                
                print("\n✅ API test completed successfully")
            else:
                print(f"❌ API test failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ API test error: {str(e)}")
            print("Make sure the API server is running")
    
    def run_email_checker(self):
        """Run the direct email checker."""
        print("\n📧 Starting direct email checker...")
        print("This will open a new process to check your emails")
        
        try:
            subprocess.run([sys.executable, 'email_integration.py'])
        except KeyboardInterrupt:
            print("\n📧 Email checker stopped")
        except Exception as e:
            print(f"❌ Email checker error: {str(e)}")
    
    def stop_all_services(self):
        """Stop all running services."""
        print("🛑 Stopping all services...")
        
        for service, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Stopped {service}")
            except:
                try:
                    process.kill()
                    print(f"🔥 Force stopped {service}")
                except:
                    print(f"⚠️  Could not stop {service}")
        
        self.processes.clear()
    
    def signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        print("\n🛑 Shutting down system...")
        self.stop_all_services()
        sys.exit(0)
    
    def run_full_system(self):
        """Run the complete spam detection system."""
        print("🚀 STARTING EMAIL SPAM DETECTION SYSTEM")
        print("="*50)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            return False
        
        # Step 2: Check/train model
        if not self.check_model_status():
            print("\n🎯 No trained model found. Starting training...")
            if not self.train_model():
                print("❌ Cannot proceed without trained model")
                return False
        
        # Step 3: Start services
        if not self.start_api_server():
            print("❌ Failed to start API server")
            return False
        
        if not self.start_email_processor():
            print("❌ Failed to start email processor")
            return False
        
        # Step 4: System ready
        self.system_ready = True
        print("\n🎉 SYSTEM READY!")
        self.show_system_status()
        
        # Step 5: Interactive mode
        try:
            self.run_interactive_mode()
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
        
        return True

def main():
    """Main entry point."""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    system = SpamDetectionSystem()
    
    print("Welcome to the Email Spam Detection System!")
    print("\nChoose startup mode:")
    print("1. Full system startup (recommended)")
    print("2. Training only")
    print("3. API server only")
    print("4. Email checker only")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        system.run_full_system()
    elif choice == '2':
        system.train_model()
    elif choice == '3':
        if system.check_dependencies():
            system.start_api_server()
            print("API server running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                system.stop_all_services()
    elif choice == '4':
        system.run_email_checker()
    else:
        print("Invalid choice. Running full system...")
        system.run_full_system()

if __name__ == "__main__":
    main()