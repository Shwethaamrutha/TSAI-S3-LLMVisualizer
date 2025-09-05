import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file if it exists
try:
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
except FileNotFoundError:
    pass  # .env file is optional

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import traceback
import logging
import json
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import requests

AVAILABLE_COLORSCALES = [
    'Viridis', 'Plasma', 'Inferno', 'Magma',  # Sequential
    'RdBu', 'RdYlBu', 'PiYG', 'PRGn',        # Diverging
    'Rainbow', 'Jet', 'Turbo',                # Spectral
    'Blues', 'Greens', 'Reds', 'Purples'      # Single color
]


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

# Initialize Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')  # Default to latest 2.5 Flash model

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use configurable Gemini model - defaults to 2.5 Flash (latest, fastest, most efficient)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    logger.info(f"Gemini API configured successfully with model: {GEMINI_MODEL_NAME}")
else:
    gemini_model = None
    logger.warning("GEMINI_API_KEY not found. Gemini features will be disabled.")

try:
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Helper functions for AI features
def get_smart_clustering_analysis(tokens, embeddings_3d):
    """Get Gemini's analysis of semantic clusters in 3D embedding space"""
    if not gemini_model:
        return "Gemini API not configured."
    
    try:
        # Create a simple representation of the 3D space
        token_positions = []
        for i, token in enumerate(tokens):  # Analyze all tokens, not just first 10
            if i < len(embeddings_3d):
                x, y, z = embeddings_3d[i][:3]
                token_positions.append(f"'{token}' at position ({x:.2f}, {y:.2f}, {z:.2f})")
        
        positions_text = "\n".join(token_positions)
        
        prompt = f"""
        Analyze the semantic clustering in this 3D embedding space and provide insights in the following exact format:
        
        Token positions in 3D space:
        {positions_text}
        
        For each semantic cluster you identify, provide the analysis in this exact format:
        
        **Cluster Name**
        **Tokens:** [list the specific tokens in this cluster]
        **Theme:** [brief explanation of the semantic theme]
        **Position:** [analysis of spatial relationships and significance]
        
        Example format:
        **Human Paradox**
        **Tokens:** humanity, compassion, paradox, cruelty
        **Theme:** Represents the contradiction between great kindness and immense cruelty in human nature
        **Position:** These tokens cluster together in the semantic space, showing their conceptual relationship
        
        IMPORTANT: 
        - Analyze ALL tokens provided, not just a subset
        - Group tokens into 4-6 meaningful semantic clusters
        - Each cluster should contain 2-8 tokens that are semantically related
        - Include special tokens like [CLS], [SEP], [UNK] in appropriate clusters
        - Ensure every token is assigned to a cluster
        - Format each cluster exactly as shown above.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error getting clustering analysis: {str(e)}")
        return f"Error generating clustering analysis: {str(e)}"

def generate_smart_clustering_analysis(tokens, embeddings_3d):
    """Generate intelligent clustering analysis using local algorithms"""
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Use all tokens for comprehensive analysis
        max_tokens = len(tokens)
        tokens_subset = tokens[:max_tokens]
        embeddings_subset = embeddings_3d[:max_tokens]
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings_subset)
        
        # Determine optimal number of clusters (4-8 for better coverage)
        n_clusters = min(8, max(4, len(tokens_subset) // 4))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # Analyze each cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((i, tokens_subset[i], embeddings_subset[i]))
        
        # Generate analysis for each cluster
        analysis_parts = []
        cluster_names = [
            "Core Semantic Group",
            "Contextual Modifiers", 
            "Structural Elements",
            "Specialized Terms",
            "Functional Tokens",
            "Semantic Modifiers",
            "Content Words",
            "Grammatical Elements"
        ]
        
        for cluster_id, cluster_data in clusters.items():
            if len(cluster_data) < 2:
                continue
                
            cluster_tokens = [item[1] for item in cluster_data]
            cluster_embeddings = [item[2] for item in cluster_data]
            
            # Calculate cluster statistics
            cluster_center = np.mean(cluster_embeddings, axis=0)
            cluster_spread = np.std(cluster_embeddings, axis=0)
            
            # Determine cluster characteristics
            if cluster_id < len(cluster_names):
                cluster_name = cluster_names[cluster_id]
            else:
                cluster_name = f"Semantic Cluster {cluster_id + 1}"
            
            # Analyze semantic theme based on token types
            theme = analyze_cluster_theme(cluster_tokens)
            
            # Analyze spatial position
            position_analysis = analyze_cluster_position(cluster_embeddings, cluster_center)
            
            analysis_parts.append(f"""**{cluster_name}**
**Tokens:** {', '.join(cluster_tokens)}
**Theme:** {theme}
**Position:** {position_analysis}""")
        
        if not analysis_parts:
            return """**General Token Distribution**
**Tokens:** All analyzed tokens
**Theme:** Individual tokens with distinct semantic positions
**Position:** Tokens are distributed across the embedding space, each representing unique semantic concepts"""
        
        return "\n\n".join(analysis_parts)
        
    except Exception as e:
        logger.error(f"Error in smart clustering analysis: {str(e)}")
        return f"Smart clustering analysis error: {str(e)}"

def analyze_cluster_theme(tokens):
    """Analyze the semantic theme of a cluster based on token characteristics"""
    # Common semantic patterns
    if any(token in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 'with'] for token in tokens):
        return "Function words and grammatical connectors that provide structural support to sentences"
    elif any(token.startswith('##') for token in tokens):
        return "Subword tokens representing morphological components of larger words"
    elif any(token in ['[CLS]', '[SEP]', '[UNK]'] for token in tokens):
        return "Special tokens used by BERT for sentence processing and tokenization"
    elif any(token.isupper() or token.istitle() for token in tokens):
        return "Proper nouns and capitalized terms representing specific entities or concepts"
    elif len(tokens) > 3:
        return "Content words representing core semantic concepts and meaning"
    else:
        return "Mixed semantic group with diverse linguistic functions"

def analyze_cluster_position(embeddings, center):
    """Analyze the spatial position and characteristics of a cluster"""
    try:
        import numpy as np
        
        # Calculate cluster spread
        spread = np.std(embeddings, axis=0)
        avg_spread = np.mean(spread)
        
        # Analyze position relative to origin
        center_magnitude = np.linalg.norm(center)
        
        if avg_spread < 20:
            spread_desc = "tightly clustered"
        elif avg_spread < 50:
            spread_desc = "moderately clustered"
        else:
            spread_desc = "loosely distributed"
        
        if center_magnitude < 50:
            position_desc = "near the semantic center"
        elif center_magnitude < 100:
            position_desc = "in the mid-range semantic space"
        else:
            position_desc = "in the outer semantic regions"
        
        return f"Tokens are {spread_desc} in the embedding space, positioned {position_desc} with center at ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
        
    except Exception as e:
        return "Spatial analysis shows these tokens cluster together in the embedding space"

def generate_fallback_clustering_analysis(tokens, embeddings_3d):
    """Fallback clustering analysis when Gemini is filtered"""
    try:
        # Simple distance-based clustering
        clusters = []
        
        # Group tokens by proximity using all tokens
        if len(tokens) >= 2:
            # Find tokens with similar Y coordinates (common semantic dimension)
            y_coords = [(i, embeddings_3d[i][1]) for i in range(len(tokens))]
            y_coords.sort(key=lambda x: x[1])
            
            # Create clusters based on Y-coordinate proximity
            current_cluster = [y_coords[0]]
            for i in range(1, len(y_coords)):
                if abs(y_coords[i][1] - current_cluster[-1][1]) < 30:  # Reduced threshold for more clusters
                    current_cluster.append(y_coords[i])
                else:
                    if len(current_cluster) >= 1:  # Allow single-token clusters
                        clusters.append(current_cluster)
                    current_cluster = [y_coords[i]]
            
            if len(current_cluster) >= 1:  # Allow single-token clusters
                clusters.append(current_cluster)
        
        # Generate analysis for each cluster
        analysis_parts = []
        for i, cluster in enumerate(clusters[:6]):  # Max 6 clusters for better coverage
            cluster_tokens = [tokens[idx] for idx, _ in cluster]
            cluster_name = f"Semantic Cluster {i+1}"
            
            analysis_parts.append(f"""**{cluster_name}**
**Tokens:** {', '.join(cluster_tokens)}
**Theme:** Tokens with similar semantic positioning in the embedding space
**Position:** These tokens cluster together based on their Y-coordinate proximity, indicating shared semantic characteristics""")
        
        if not analysis_parts:
            return """**General Token Distribution**
**Tokens:** All analyzed tokens
**Theme:** Individual tokens with distinct semantic positions
**Position:** Tokens are distributed across the embedding space, each representing unique semantic concepts"""
        
        return "\n\n".join(analysis_parts)
    except Exception as e:
        return f"Fallback analysis error: {str(e)}"

def get_attention_pattern_analysis(tokens, attention_scores):
    """Get Gemini's analysis of attention patterns from the specific visualization"""
    if not gemini_model:
        return "Gemini API not configured."
    
    try:
        # Analyze the specific attention matrix for this text
        num_tokens = len(tokens)
        max_attention = max(max(row) for row in attention_scores)
        min_attention = min(min(row) for row in attention_scores)
        
        # Find the strongest attention patterns
        strong_patterns = []
        self_attention_strengths = []
        
        for i in range(num_tokens):
            for j in range(num_tokens):
                if i != j and attention_scores[i][j] > 0.08:  # Lower threshold to capture cross-attention patterns visible in heatmap
                    strong_patterns.append(f"'{tokens[i]}' → '{tokens[j]}' ({attention_scores[i][j]:.3f})")
                elif i == j:
                    self_attention_strengths.append(f"'{tokens[i]}' self-attention: {attention_scores[i][j]:.3f}")
        
        # Get top patterns - sort by attention score to get the strongest ones
        strong_patterns.sort(key=lambda x: float(x.split('(')[1].split(')')[0]), reverse=True)
        top_patterns = strong_patterns[:12]  # Top 12 cross-attention patterns
        top_self_attention = sorted(self_attention_strengths, key=lambda x: float(x.split(': ')[1]), reverse=True)[:5]
        
        patterns_text = "\n".join(top_patterns) if top_patterns else "No strong cross-attention patterns found"
        self_attention_text = "\n".join(top_self_attention)
        
        prompt = f"""
        Analyze this attention matrix visualization for the text with {num_tokens} tokens and provide insights in the following exact format:
        
        **Attention Matrix Statistics:**
        - Matrix size: {num_tokens} × {num_tokens}
        - Max attention value: {max_attention:.3f}
        - Min attention value: {min_attention:.3f}
        
        **Strongest Cross-Attention Patterns:**
        {patterns_text}
        
        **Strongest Self-Attention (diagonal):**
        {self_attention_text}
        
        For each attention pattern you identify, provide the analysis in this exact format:
        
        **Pattern Name**
        **Tokens:** [list the specific tokens involved in this pattern]
        **Theme:** [brief explanation of the linguistic or semantic relationship]
        **Position:** [analysis of attention strength and significance in the text]
        
        Example format:
        **Subject-Verb Relationships**
        **Tokens:** "The", "cat", "sleeps", "peacefully"
        **Theme:** Shows strong attention between subject and verb, indicating grammatical dependency
        **Position:** High attention values (0.8+) between "cat" and "sleeps" demonstrate core syntactic relationships
        
        Identify 3-5 distinct attention patterns and format each one exactly as shown above.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error getting attention pattern analysis: {str(e)}")
        return f"Error generating attention pattern analysis: {str(e)}"

def generate_smart_attention_analysis(tokens, attention_scores):
    """Generate intelligent attention pattern analysis using local algorithms"""
    try:
        import numpy as np
        
        # Limit to reasonable number of tokens
        max_tokens = min(8, len(tokens))
        tokens_subset = tokens[:max_tokens]
        attention_subset = np.array(attention_scores[:max_tokens][:max_tokens])
        
        # Find strongest attention patterns
        patterns = []
        
        # Self-attention analysis (diagonal)
        self_attention = np.diag(attention_subset)
        strong_self_indices = np.where(self_attention > 0.3)[0]
        
        # Cross-attention analysis (off-diagonal)
        cross_attention = attention_subset.copy()
        np.fill_diagonal(cross_attention, 0)  # Remove diagonal
        strong_cross_indices = np.where(cross_attention > 0.2)
        
        # Analyze self-attention patterns
        if len(strong_self_indices) > 0:
            for idx in strong_self_indices[:2]:  # Top 2 self-attention
                token = tokens_subset[idx]
                strength = self_attention[idx]
                pattern_name = f"Self-Attention Focus"
                theme = analyze_self_attention_theme(token, strength)
                position = f"High self-attention value ({strength:.3f}) indicates strong focus on '{token}'"
                
                patterns.append(f"""**{pattern_name}**
**Tokens:** {token}
**Theme:** {theme}
**Position:** {position}""")
        
        # Analyze cross-attention patterns
        if len(strong_cross_indices[0]) > 0:
            # Get top cross-attention pairs
            cross_pairs = list(zip(strong_cross_indices[0], strong_cross_indices[1]))
            cross_strengths = [cross_attention[i, j] for i, j in cross_pairs]
            
            # Sort by strength
            sorted_pairs = sorted(zip(cross_pairs, cross_strengths), key=lambda x: x[1], reverse=True)
            
            for (i, j), strength in sorted_pairs[:3]:  # Top 3 cross-attention
                token1, token2 = tokens_subset[i], tokens_subset[j]
                pattern_name = f"Cross-Attention Pattern"
                theme = analyze_cross_attention_theme(token1, token2, strength)
                position = f"Strong attention ({strength:.3f}) from '{token1}' to '{token2}'"
                
                patterns.append(f"""**{pattern_name}**
**Tokens:** {token1}, {token2}
**Theme:** {theme}
**Position:** {position}""")
        
        if not patterns:
            return """**General Attention Distribution**
**Tokens:** All analyzed tokens
**Theme:** Standard attention patterns across the sequence
**Position:** Attention values show typical token relationships in the text"""
        
        return "\n\n".join(patterns)
        
    except Exception as e:
        logger.error(f"Error in smart attention analysis: {str(e)}")
        return f"Smart attention analysis error: {str(e)}"

def analyze_self_attention_theme(token, strength):
    """Analyze the theme of self-attention for a token"""
    if token in ['[CLS]', '[SEP]']:
        return "Special token with high self-attention indicating important sentence-level processing"
    elif token.startswith('##'):
        return "Subword token with strong self-focus, likely part of a complex word"
    elif strength > 0.5:
        return "High self-attention indicating this token is central to the sentence meaning"
    else:
        return "Moderate self-attention showing this token has some individual significance"

def analyze_cross_attention_theme(token1, token2, strength):
    """Analyze the theme of cross-attention between two tokens"""
    # Check for grammatical relationships
    if token1.lower() in ['the', 'a', 'an'] and not token2.startswith('##'):
        return "Determiner-noun relationship showing grammatical dependency"
    elif token1.startswith('##') and not token2.startswith('##'):
        return "Subword-to-word connection indicating morphological relationship"
    elif token1 in ['[CLS]', '[SEP]'] or token2 in ['[CLS]', '[SEP]']:
        return "Special token attention indicating sentence-level processing"
    elif strength > 0.4:
        return "Strong semantic relationship between these tokens"
    else:
        return "Moderate attention relationship showing linguistic connection"

def generate_fallback_attention_analysis(tokens, attention_scores):
    """Fallback attention analysis when Gemini is filtered"""
    try:
        patterns = []
        
        # Find strongest attention patterns
        max_attention = 0
        strong_pairs = []
        
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if i != j and attention_scores[i][j] > 0.2:
                    strong_pairs.append((i, j, attention_scores[i][j]))
                    max_attention = max(max_attention, attention_scores[i][j])
        
        # Sort by attention strength
        strong_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Generate analysis for top patterns
        for i, (idx1, idx2, strength) in enumerate(strong_pairs[:3]):
            pattern_name = f"Attention Pattern {i+1}"
            tokens_involved = f"{tokens[idx1]}, {tokens[idx2]}"
            
            patterns.append(f"""**{pattern_name}**
**Tokens:** {tokens_involved}
**Theme:** Strong attention relationship between these tokens
**Position:** High attention value ({strength:.3f}) indicates significant linguistic connection""")
        
        if not patterns:
            return """**General Attention Distribution**
**Tokens:** All analyzed tokens
**Theme:** Standard attention patterns across the sequence
**Position:** Attention values show typical token relationships in the text"""
        
        return "\n\n".join(patterns)
    except Exception as e:
        return f"Fallback attention analysis error: {str(e)}"

def generate_sample_text(topic, complexity):
    """Generate sample text for testing using Gemini"""
    if not gemini_model:
        return "Gemini API not configured."
    
    try:
        complexity_descriptions = {
            "simple": "simple, straightforward language with basic sentence structures",
            "medium": "moderate complexity with some compound sentences and varied vocabulary",
            "complex": "highly complex with intricate grammatical structures, technical terms, and sophisticated language"
        }
        
        complexity_desc = complexity_descriptions.get(complexity, "moderate complexity")
        
        prompt = f"""
        Generate a single sample text for BERT visualization and analysis with the following specifications:
        
        Topic: {topic}
        Complexity: {complexity_desc}
        
        Requirements:
        - 2-3 sentences long
        - Interesting for NLP analysis (good for tokenization, embeddings, and attention patterns)
        - Demonstrates linguistic phenomena relevant to the topic
        - Avoid overly long sentences that would be hard to visualize
        
        Return only the text content, no JSON formatting, no quotes, no additional commentary.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating sample text: {str(e)}")
        return f"Error generating sample text: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-sample', methods=['POST'])
def generate_sample():
    """Generate sample text for testing"""
    try:
        data = request.json
        topic = data.get('topic', 'general text')
        complexity = data.get('complexity', 'medium')
        
        sample_text = generate_sample_text(topic, complexity)
        return jsonify({'sample_text': sample_text})
    except Exception as e:
        logger.error(f"Error generating sample: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-clustering', methods=['POST'])
def analyze_clustering():
    """Analyze semantic clustering in 3D space"""
    try:
        data = request.json
        tokens = data.get('tokens', [])
        embeddings_3d = data.get('embeddings_3d', [])
        
        if not tokens or not embeddings_3d:
            return jsonify({'error': 'Missing tokens or embeddings data'}), 400
        
        analysis = get_smart_clustering_analysis(tokens, embeddings_3d)
        return jsonify({'clustering_analysis': analysis})
    except Exception as e:
        logger.error(f"Error analyzing clustering: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-attention-patterns', methods=['POST'])
def analyze_attention_patterns():
    """Analyze attention patterns"""
    try:
        data = request.json
        tokens = data.get('tokens', [])
        attention_scores = data.get('attention_scores', [])
        
        if not tokens or not attention_scores:
            return jsonify({'error': 'Missing tokens or attention scores data'}), 400
        
        analysis = get_attention_pattern_analysis(tokens, attention_scores)
        return jsonify({'attention_analysis': analysis})
    except Exception as e:
        logger.error(f"Error analyzing attention patterns: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        text = data.get('text', '')
        include_gemini = data.get('include_gemini', False)
        logger.debug(f"Received text: {text}, include_gemini: {include_gemini}")

        if not text:
            raise ValueError("No text provided")

        # Get full tokenization details
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        # Get tokens with special tokens
        full_tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        # Get embeddings and attention
        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True)
        
        embeddings = outputs.last_hidden_state.squeeze().numpy()
        
        # Calculate appropriate perplexity
        n_samples = len(full_tokens)
        perplexity = min(30, max(5, n_samples - 1))
        logger.debug(f"Using perplexity: {perplexity}")

        # 3D dimensionality reduction
        if n_samples >= 4:
            tsne = TSNE(
                n_components=3,
                random_state=42,
                perplexity=perplexity,
                n_iter_without_progress=300,
                min_grad_norm=1e-7,
            )
            embeddings_3d = tsne.fit_transform(embeddings)
            logger.debug(f"TSNE reduction completed. Shape: {embeddings_3d.shape}")
        else:
            pca = PCA(n_components=min(3, n_samples))
            embeddings_3d = pca.fit_transform(embeddings)
            if embeddings_3d.shape[1] < 3:
                padding = np.zeros((embeddings_3d.shape[0], 3 - embeddings_3d.shape[1]))
                embeddings_3d = np.hstack((embeddings_3d, padding))
            logger.debug(f"PCA reduction completed. Shape: {embeddings_3d.shape}")
        
        # Validate embeddings_3d
        logger.debug(f"Final embeddings_3d shape: {embeddings_3d.shape}")
        logger.debug(f"Embeddings_3d contains NaN: {np.isnan(embeddings_3d).any()}")
        logger.debug(f"Embeddings_3d contains Inf: {np.isinf(embeddings_3d).any()}")
        logger.debug(f"Embeddings_3d range: [{embeddings_3d.min():.3f}, {embeddings_3d.max():.3f}]")

        # Prepare token types and markers
        token_types = [
            'Special Token' if t in ['[CLS]', '[SEP]'] else
            'Subword' if t.startswith('##') else
            'Word' for t in full_tokens
        ]

        marker_symbols = [
            'circle' if not t.startswith('##') else 'diamond'
            for t in full_tokens
        ]

        # Normalize coordinates to a reasonable range for better visualization
        coords = embeddings_3d.astype(np.float64)  # Convert to float64 for better JSON serialization
        coord_range = max(abs(coords.min()), abs(coords.max()))
        if coord_range > 10:  # Only normalize if coordinates are too large
            coords = coords / coord_range * 10
            logger.debug(f"Normalized coordinates to range [-10, 10]")
        
        # Convert to regular Python lists with float values
        x_coords = [float(x) for x in coords[:, 0]]
        y_coords = [float(y) for y in coords[:, 1]]
        z_coords = [float(z) for z in coords[:, 2]]
        
        # Create scatter data for 3D visualization
        scatter_data = {
            'data': [{
                'type': 'scatter3d',
                'x': x_coords,
                'y': y_coords,
                'z': z_coords,
                'mode': 'markers+text',  # Add text labels to the markers
                'text': full_tokens,  # Token names for labels
                'textposition': 'top center',  # Position text above the points
                'textfont': {
                    'size': 8,  # Smaller font size to avoid overlap
                    'color': 'black'
                },
                'textangle': 0,  # No rotation for better readability
                'marker': {
                    'size': 8,  # Increased size for better visibility
                    'color': list(range(len(full_tokens))),
                    'colorscale': 'Viridis',
                    'opacity': 0.9,
                    'showscale': True,
                    'colorbar': {
                        'title': 'Token Position',
                        'thickness': 15,
                        'len': 0.5,
                        'x': 0.85
                    },
                    'line': {
                        'color': 'black',
                        'width': 1
                    }
                },
                'hovertemplate': 
                    '<b>Token:</b> %{text}<br>' +
                    '<b>Position:</b> %{marker.color}<br>' +
                    '<b>X:</b> %{x:.3f}<br>' +
                    '<b>Y:</b> %{y:.3f}<br>' +
                    '<b>Z:</b> %{z:.3f}<br>' +
                    '<extra></extra>'
            }],
            'layout': {
                'title': {
                    'text': '3D Token Embeddings',
                    'font': {'size': 16}
                },
                'scene': {
                    'xaxis': {
                        'title': 'X',
                        'showgrid': True,
                        'zeroline': True,
                        'range': [float(coords[:, 0].min() - 1), float(coords[:, 0].max() + 1)]
                    },
                    'yaxis': {
                        'title': 'Y',
                        'showgrid': True,
                        'zeroline': True,
                        'range': [float(coords[:, 1].min() - 1), float(coords[:, 1].max() + 1)]
                    },
                    'zaxis': {
                        'title': 'Z',
                        'showgrid': True,
                        'zeroline': True,
                        'range': [float(coords[:, 2].min() - 1), float(coords[:, 2].max() + 1)]
                    },
                    'camera': {
                        'up': {'x': 0, 'y': 0, 'z': 1},
                        'center': {'x': 0, 'y': 0, 'z': 0},
                        'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
                    },
                    'aspectmode': 'cube'
                },
                'margin': {'l': 0, 'r': 0, 'b': 0, 't': 50},
                'showlegend': False,
                'width': 800,
                'height': 600
            }
        }



        # Get attention scores - use the last layer for better attention patterns
        attention_scores = outputs.attentions[-1].mean(dim=1)[0].numpy()
        
        # Normalize attention scores for better visualization
        # Apply softmax to make the attention distribution more pronounced
        attention_scores_softmax = np.exp(attention_scores * 10) / np.sum(np.exp(attention_scores * 10), axis=-1, keepdims=True)
        
        # Use a combination of raw and softmax scores for better visualization
        attention_scores_final = 0.7 * attention_scores + 0.3 * attention_scores_softmax
        
        # Debug logging
        logger.debug(f"Scatter data structure: {scatter_data}")
        logger.debug(f"Embeddings 3D shape: {embeddings_3d.shape}")
        logger.debug(f"Number of tokens: {len(full_tokens)}")
        logger.debug(f"X coordinates: {embeddings_3d[:, 0].tolist()[:5]}...")  # First 5 values
        logger.debug(f"Y coordinates: {embeddings_3d[:, 1].tolist()[:5]}...")  # First 5 values
        logger.debug(f"Z coordinates: {embeddings_3d[:, 2].tolist()[:5]}...")  # First 5 values
        
        # Attention scores debugging
        logger.debug(f"Attention scores shape: {attention_scores.shape}")
        logger.debug(f"Raw attention scores range: [{attention_scores.min():.4f}, {attention_scores.max():.4f}]")
        logger.debug(f"Final attention scores range: [{attention_scores_final.min():.4f}, {attention_scores_final.max():.4f}]")
        logger.debug(f"Self-attention diagonal: {np.diag(attention_scores_final).tolist()}")
        
        # Prepare BERT insights for Gemini analysis
        bert_insights = {
            'num_tokens': len(full_tokens),
            'embedding_shape': embeddings.shape,
            'attention_info': f"Max attention: {attention_scores_final.max():.4f}, Min attention: {attention_scores_final.min():.4f}"
        }
        

        # Prepare response data
        response_data = {
            'tokens': full_tokens[1:-1],
            'token_ids': encoded['input_ids'][0].tolist(),
            'embeddings': [[float(val) for val in row] for row in embeddings_3d.tolist()],
            'original_embeddings': [[float(val) for val in row] for row in embeddings.tolist()],
            'attention_scores': [[float(val) for val in row] for row in attention_scores_final.tolist()],
            'plot_data': scatter_data,
            'technical_info': {
                'input_text_length': len(text),
                'num_tokens': len(full_tokens),
                'embedding_shape': embeddings.shape,
                'perplexity': perplexity,
                'model_name': model_name,
                'vocab_size': tokenizer.vocab_size,
                'hidden_size': embeddings.shape[-1],
                'num_attention_heads': model.config.num_attention_heads,
                'num_hidden_layers': model.config.num_hidden_layers
            },
        }
        response_data['available_colorscales'] = AVAILABLE_COLORSCALES
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
