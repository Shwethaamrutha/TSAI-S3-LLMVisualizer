# LLM Visualizer with Gemini Integration

A web-based tool for visualizing how BERT processes text with AI-powered analysis using Google's Gemini model.

## 🚀 Features

- **3D Embedding Visualization**: Interactive 3D scatter plot of token embeddings
- **Attention Matrix Analysis**: Heatmap showing attention patterns between tokens
- **AI-Powered Analysis**: Smart clustering and attention pattern recognition using Gemini 2.5 Flash
- **Sample Text Generation**: Generate test texts with customizable topics and complexity
- **Interactive Highlighting**: Click clusters to highlight tokens in 3D plot
- **Fullscreen Plots**: Enlarge visualizations for detailed exploration
- **Step-by-Step Tokenization**: Visual breakdown of the tokenization process

## 🛠️ Technology Stack

- **Backend**: Python 3.12+, Flask, Transformers, PyTorch, Google Generative AI
- **Frontend**: HTML5, CSS3, JavaScript, Plotly.js, D3.js
- **Package Manager**: uv (recommended) or pip

## 📋 Prerequisites

- Python 3.12 or higher
- Google AI Studio API key (for Gemini features)
- Modern web browser with JavaScript enabled

## 🚀 Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/Shwethaamrutha/TSAI-S2.git
cd TSAI-S2

# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies
uv sync
```

### 2. Setup Gemini API
```bash
# Run setup script
python setup_gemini.py

# Or manually set API key
export GEMINI_API_KEY="your_api_key_here"
# Get API key from: https://makersuite.google.com/app/apikey
```

### 3. Run Application
```bash
uv run python app.py
```
Access at: `http://localhost:5000`

## 📖 Usage

1. **Enter text** in the input field
2. **Click "Generate"** to create sample text (optional)
3. **Click "Process"** to generate visualizations
4. **Click "Analyze Clusters"** for semantic clustering analysis
5. **Click "Analyze Patterns"** for attention pattern analysis
6. **Click clusters/patterns** to highlight them in the plots

## 🚀 Deployment

### Local Development
```bash
uv run python app.py
# Or: python app.py
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Environment variables
export GEMINI_API_KEY="your_api_key_here"
export GEMINI_MODEL="gemini-2.5-flash"
```

### Docker Deployment
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Disk Space Issues (EC2)
```bash
# Check space
df -h

# Clear caches
rm -rf ~/.cache/uv/
pip cache purge

# Use CPU-only PyTorch
pip install -r requirements-cpu.txt
```

#### 2. Build Issues
```bash
# If uv sync fails, use pip
pip install -r requirements.txt

# Or install individually
pip install flask transformers torch numpy scikit-learn plotly google-generativeai requests
```

#### 3. Permission Issues
```bash
# Try different port
python -c "from app import app; app.run(debug=True, host='0.0.0.0', port=8080)"
```

#### 4. Gemini API Issues
```bash
# Check API key
echo $GEMINI_API_KEY

# Set manually
export GEMINI_API_KEY="your_api_key_here"

# Test connectivity
curl -H "Content-Type: application/json" \
     -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
     "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=YOUR_API_KEY"
```

### EC2 Deployment

#### Method 1: Using uv (Recommended)
```bash
# 1. Check disk space
df -h

# 2. Move to EBS volume if needed
cd /mnt/data
git clone <your-repo>
cd TSAI-S2

# 3. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 4. Set Hugging Face cache to EBS volume
export HF_HOME=/mnt/data/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/data/huggingface_cache
export HF_DATASETS_CACHE=/mnt/data/huggingface_cache

# 5. Make environment variables permanent
sudo tee /etc/profile.d/huggingface.sh << EOF
export HF_HOME=/mnt/data/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/data/huggingface_cache
export HF_DATASETS_CACHE=/mnt/data/huggingface_cache
EOF

# 6. Install and run
uv sync
python setup_gemini.py
uv run python app.py
```

#### Method 2: Using pip (Alternative)
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install CPU-only PyTorch
pip install -r requirements-cpu.txt

# 3. Set cache locations (same as above)
export HF_HOME=/mnt/data/huggingface_cache
# ... (same environment setup)

# 4. Setup and run
python setup_gemini.py
python app.py
```

### Environment Variables
```bash
# Required
export GEMINI_API_KEY="your_api_key_here"
export GEMINI_MODEL="gemini-2.5-flash"

# Optional (for cache management)
export HF_HOME="/path/to/cache"
export TRANSFORMERS_CACHE="/path/to/cache"
```

### API Quota Management
- **Free Tier**: Limited requests per day
- **Paid Tier**: Higher limits, consider for production
- **Monitor Usage**: Check Google AI Studio dashboard
- **Rate Limiting**: Implement if needed for production

## 📁 Project Structure
```
TSAI-S2/
├── app.py                 # Main Flask application
├── setup_gemini.py       # Gemini API setup
├── requirements.txt      # Dependencies
├── requirements-cpu.txt  # CPU-only dependencies
├── pyproject.toml        # Project configuration
├── static/
│   ├── css/style.css     # Stylesheets
│   └── js/visualization.js # Frontend logic
└── templates/
    └── index.html        # Main template
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for Transformers library and BERT model
- **Google AI** for Gemini API
- **Plotly** for visualization library

---

**Built for AI interpretability and educationi