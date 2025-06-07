#!/usr/bin/env python3
"""
Quick Setup Script for Email Spam Detection System
Run this first to ensure everything is properly configured
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install all required packages."""
    print("📦 Installing required packages...")
    
    requirements = [
        "transformers==4.30.2",
        "torch==2.0.1", 
        "pandas==2.0.3",
        "numpy==1.25.1",
        "datasets==2.13.1",
        "scikit-learn==1.3.0",
        "flask==2.3.2",
        "tqdm==4.65.0",
        "accelerate>=0.20.1",
        "requests==2.31.0"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def create_requirements_file():
    """Create requirements.txt file."""
    requirements_content = """transformers==4.30.2
torch==2.0.1
pandas==2.0.3
numpy==1.25.1
datasets==2.13.1
scikit-learn==1.3.0
flask==2.3.2
tqdm==4.65.0
accelerate>=0.20.1
requests==2.31.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    print("✅ requirements.txt created")

def check_files():
    """Check if all required files exist."""
    required_files = [
        'spam_detec.py',
        'run_training.py', 
        'run_api.py',
        'email_integration.py',
        'email_processor.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        print("Please ensure all files are in the current directory")
        return False
    
    print("✅ All required files present")
    return True

def main():
    print("🚀 Email Spam Detection System - Quick Setup")
    print("="*50)
    
    # Check files
    if not check_files():
        return
    
    # Create requirements file
    create_requirements_file()
    
    # Install packages
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python master_startup.py")
    print("2. Choose option 1 for full system startup")
    print("3. Follow the interactive prompts")
    
    print("\n📚 Quick Start Guide:")
    print("• Full system: python master_startup.py")
    print("• Training only: python run_training.py") 
    print("• API only: python run_api.py")
    print("• Check emails: python email_integration.py")

if __name__ == "__main__":
    main()