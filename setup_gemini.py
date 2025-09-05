#!/usr/bin/env python3
"""
Setup script for Gemini API integration
This script helps configure the Gemini API key for the LLM Visualizer
"""

import os
import sys

def setup_gemini_api():
    """Setup Gemini API key"""
    print("🤖 Gemini API Setup for LLM Visualizer")
    print("=" * 50)
    print("📋 Available Gemini Models:")
    print("  • gemini-2.5-flash (Default) - Latest 2.5 Flash, fastest & most advanced")
    print("  • gemini-1.5-flash - Previous generation, fast & efficient")
    print("  • gemini-1.5-pro - Most capable, higher quality responses")
    print("  • gemini-pro - Original model (legacy)")
    print("=" * 50)
    
    # Check if API key is already set
    current_key = os.getenv('GEMINI_API_KEY')
    if current_key:
        print(f"✅ GEMINI_API_KEY is already set: {current_key[:10]}...")
        choice = input("Do you want to update it? (y/n): ").lower()
        if choice != 'y':
            print("Keeping existing API key.")
            return
    
    print("\n📋 To get your Gemini API key:")
    print("1. Visit: https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated API key")
    
    api_key = input("\n🔑 Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    # Ask for model preference
    print("\n🤖 Choose Gemini Model:")
    print("1. gemini-2.5-flash (Recommended - Latest & Most Advanced)")
    print("2. gemini-1.5-flash (Fast & Efficient)")
    print("3. gemini-1.5-pro (Most Capable)")
    print("4. gemini-pro (Legacy)")
    
    model_choice = input("Enter choice (1-4) or press Enter for default: ").strip()
    
    model_map = {
        '1': 'gemini-2.5-flash',
        '2': 'gemini-1.5-flash',
        '3': 'gemini-1.5-pro', 
        '4': 'gemini-pro'
    }
    
    selected_model = model_map.get(model_choice, 'gemini-2.5-flash')
    print(f"✅ Selected model: {selected_model}")
    
    # Validate API key format (basic check)
    if not api_key.startswith('AIza'):
        print("⚠️  Warning: API key doesn't start with 'AIza'. Please verify it's correct.")
        confirm = input("Continue anyway? (y/n): ").lower()
        if confirm != 'y':
            return
    
    # Set environment variable for current session
    os.environ['GEMINI_API_KEY'] = api_key
    
    # Create .env file for persistence
    env_content = f"GEMINI_API_KEY={api_key}\nGEMINI_MODEL={selected_model}\n"
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ API key saved to .env file")
    except Exception as e:
        print(f"⚠️  Could not save to .env file: {e}")
        print("You can manually set the environment variables:")
        print(f"export GEMINI_API_KEY={api_key}")
        print(f"export GEMINI_MODEL={selected_model}")
    
    print("\n🚀 Setup complete! You can now run the application with Gemini support.")
    print("To test the integration, run: python app.py")

def load_env_file():
    """Load environment variables from .env file"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Loaded environment variables from .env file")
    except FileNotFoundError:
        print("ℹ️  No .env file found. You can create one with your API keys.")
    except Exception as e:
        print(f"⚠️  Error loading .env file: {e}")

if __name__ == "__main__":
    load_env_file()
    setup_gemini_api()
