#!/usr/bin/env python3
"""
Demo script showing Gemini integration features
This script demonstrates the new LLM capabilities added to the BERT Visualizer
"""

import os
import sys
import json
from app import get_gemini_analysis, get_gemini_embedding, compare_embeddings
import numpy as np

def demo_gemini_features():
    """Demonstrate Gemini integration features"""
    print("🤖 Gemini Integration Demo")
    print("=" * 50)
    
    # Check if Gemini is available
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY not found!")
        print("Please run: python setup_gemini.py")
        return
    
    # Sample text for analysis
    sample_text = "Cats and dogs, humanity's two most popular domestic companions, offer distinct types of companionship."
    
    print(f"📝 Sample Text: {sample_text}")
    print("\n" + "="*50)
    
    # Simulate BERT insights (normally from the actual BERT processing)
    bert_insights = {
        'num_tokens': 15,
        'embedding_shape': (15, 768),
        'attention_info': 'Max attention: 0.8234, Min attention: 0.0123'
    }
    
    print("🔍 Getting Gemini Analysis...")
    try:
        # Get AI analysis
        analysis = get_gemini_analysis(sample_text, bert_insights)
        print("✅ Gemini Analysis:")
        print("-" * 30)
        print(analysis)
        print("\n" + "="*50)
        
        # Get Gemini embedding (simulated)
        print("🧠 Getting Gemini Embedding...")
        embedding, response = get_gemini_embedding(sample_text)
        
        if embedding is not None:
            print("✅ Gemini Embedding Generated")
            print(f"   - Dimension: {len(embedding)}")
            print(f"   - Sample values: {embedding[:5]}")
            print(f"   - Response: {response[:100]}...")
            
            # Simulate BERT embeddings for comparison
            bert_embeddings = np.random.normal(0, 1, (15, 768))
            
            print("\n📊 Comparing Embeddings...")
            comparison = compare_embeddings(bert_embeddings, [embedding])
            
            if comparison:
                print("✅ Embedding Comparison:")
                print(f"   - Cosine Similarity: {comparison['cosine_similarity']:.4f}")
                print(f"   - BERT Magnitude: {comparison['bert_mean_magnitude']:.4f}")
                print(f"   - Gemini Magnitude: {comparison['gemini_mean_magnitude']:.4f}")
                print(f"   - Dimension Difference: {comparison['dimensionality_difference']}")
        
        print("\n🎉 Demo completed successfully!")
        print("\nTo see these features in action:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
        print("3. Enable 'Include Gemini Analysis' checkbox")
        print("4. Enter text and click 'Process'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your GEMINI_API_KEY is valid and you have internet access.")

if __name__ == "__main__":
    demo_gemini_features()
