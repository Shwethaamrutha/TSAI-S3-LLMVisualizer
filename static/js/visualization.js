document.addEventListener('DOMContentLoaded', function() {
    const processBtn = document.getElementById('process-btn');
    const inputText = document.getElementById('input-text');
    
    // Verify 3D plot container exists
    const embeddingPlot = document.getElementById('embedding-plot');
    if (!embeddingPlot) {
        console.error('3D plot container not found!');
    } else {
        console.log('3D plot container found and ready');
    }
    
    // Verify Plotly is loaded
    if (typeof Plotly === 'undefined') {
        console.error('Plotly is not loaded!');
    } else {
        console.log('Plotly is loaded successfully, version:', Plotly.version);
    }
    
    // Add event listener for text labels toggle
    const textLabelsCheckbox = document.getElementById('show-text-labels');
    if (textLabelsCheckbox) {
        textLabelsCheckbox.addEventListener('change', function() {
            const showLabels = this.checked;
            const plotElement = document.getElementById('embedding-plot');
            
            if (plotElement && plotElement.data) {
                Plotly.restyle('embedding-plot', {
                    'mode': showLabels ? 'markers+text' : 'markers'
                });
            }
        });
    }
    
    // Add event listener for sample generation
    const generateSampleBtn = document.getElementById('generate-sample-btn');
    if (generateSampleBtn) {
        generateSampleBtn.addEventListener('click', function() {
            showSampleGeneratorModal();
        });
    }
    
    // Add event listeners for analysis buttons
    const analyzeClusteringBtn = document.getElementById('analyze-clustering-btn');
    if (analyzeClusteringBtn) {
        analyzeClusteringBtn.addEventListener('click', function() {
            if (!processData || !processData.tokens || !processData.embeddings) {
                alert('Please process some text first to analyze clustering.');
                return;
            }
            
            analyzeClusteringBtn.disabled = true;
            analyzeClusteringBtn.textContent = 'Analyzing...';
            
            fetch('/analyze-clustering', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    tokens: processData.tokens,
                    embeddings_3d: processData.embeddings
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.clustering_analysis) {
                    updateClusteringAnalysis(data.clustering_analysis);
                } else {
                    alert('Error analyzing clustering: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error analyzing clustering: ' + error.message);
            })
            .finally(() => {
                analyzeClusteringBtn.disabled = false;
                analyzeClusteringBtn.textContent = 'Analyze Clusters';
            });
        });
    }
    
    const analyzeAttentionBtn = document.getElementById('analyze-attention-btn');
    if (analyzeAttentionBtn) {
        analyzeAttentionBtn.addEventListener('click', function() {
            if (!processData || !processData.tokens || !processData.attention_scores) {
                alert('Please process some text first to analyze attention patterns.');
                return;
            }
            
            analyzeAttentionBtn.disabled = true;
            analyzeAttentionBtn.textContent = 'Analyzing...';
            
            fetch('/analyze-attention-patterns', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    tokens: processData.tokens,
                    attention_scores: processData.attention_scores
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.attention_analysis) {
                    updateAttentionAnalysis(data.attention_analysis);
                } else {
                    alert('Error analyzing attention patterns: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error analyzing attention patterns: ' + error.message);
            })
            .finally(() => {
                analyzeAttentionBtn.disabled = false;
                analyzeAttentionBtn.textContent = 'Analyze Patterns';
            });
        });
    }
    
    // Add event listener for attention values toggle
    const attentionValuesCheckbox = document.getElementById('show-attention-values');
    if (attentionValuesCheckbox) {
        attentionValuesCheckbox.addEventListener('change', function() {
            // Recreate the attention heatmap with updated annotation settings
            if (processData && processData.attention_scores && processData.tokens) {
                updateAttentionHeatmap(processData.attention_scores, processData.tokens);
            }
        });
    }
    
    // Download functionality
    const downloadEmbeddingsBtn = document.getElementById('download-embeddings');
    const downloadEmbeddingsJsonBtn = document.getElementById('download-embeddings-json');
    
    if (downloadEmbeddingsBtn) {
        downloadEmbeddingsBtn.addEventListener('click', function() {
            if (processData && processData.embeddings && processData.tokens) {
                downloadEmbeddingsCSV(processData.tokens, processData.embeddings);
            } else {
                alert('No embedding data available. Please process some text first.');
            }
        });
    }
    
    if (downloadEmbeddingsJsonBtn) {
        downloadEmbeddingsJsonBtn.addEventListener('click', function() {
            if (processData && processData.embeddings && processData.tokens) {
                downloadEmbeddingsJSON(processData.tokens, processData.embeddings, processData);
            } else {
                alert('No embedding data available. Please process some text first.');
            }
        });
    }
    
    function downloadEmbeddingsCSV(tokens, embeddings) {
        // Create CSV content
        let csvContent = 'Token,Position,X,Y,Z\n';
        
        tokens.forEach((token, index) => {
            const embedding = embeddings[index];
            if (embedding && embedding.length >= 3) {
                csvContent += `"${token}",${index},${embedding[0]},${embedding[1]},${embedding[2]}\n`;
            }
        });
        
        // Create and download file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'token_embeddings_3d.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    function downloadEmbeddingsJSON(tokens, embeddings, fullData) {
        // Create comprehensive JSON data
        const jsonData = {
            metadata: {
                timestamp: new Date().toISOString(),
                model: fullData.technical_info?.model_name || 'bert-base-uncased',
                total_tokens: tokens.length,
                original_dimensions: fullData.technical_info?.embedding_shape?.[1] || 'unknown',
                reduced_dimensions: 3
            },
            embeddings: tokens.map((token, index) => ({
                token: token,
                position: index,
                token_id: fullData.token_ids?.[index] || null,
                embedding_3d: embeddings[index] || [],
                embedding_full: fullData.original_embeddings?.[index] || null
            }))
        };
        
        // Create and download file
        const jsonContent = JSON.stringify(jsonData, null, 2);
        const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'token_embeddings_complete.json');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    // Tokenization process step controls
    let currentStep = 1;
    const totalSteps = 4;
    let processData = null;
    
    const prevBtn = document.getElementById('prev-step');
    const nextBtn = document.getElementById('next-step');
    const currentStepSpan = document.getElementById('current-step');
    const totalStepsSpan = document.getElementById('total-steps');
    
    if (prevBtn && nextBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentStep > 1) {
                currentStep--;
                updateStepDisplay();
            }
        });
        
        nextBtn.addEventListener('click', () => {
            if (currentStep < totalSteps) {
                currentStep++;
                updateStepDisplay();
            }
        });
    }
    
    function updateStepDisplay() {
        // Update step indicator
        currentStepSpan.textContent = currentStep;
        totalStepsSpan.textContent = totalSteps;
        
        // Update button states
        prevBtn.disabled = currentStep === 1;
        nextBtn.disabled = currentStep === totalSteps;
        
        // Hide all step content
        for (let i = 1; i <= totalSteps; i++) {
            const stepContent = document.getElementById(`step-${i}`);
            if (stepContent) {
                stepContent.style.display = 'none';
            }
        }
        
        // Show current step content
        const currentStepContent = document.getElementById(`step-${currentStep}`);
        if (currentStepContent) {
            currentStepContent.style.display = 'block';
        }
        
        // Update step content based on current step
        if (processData) {
            updateStepContent(currentStep, processData);
        }
    }
    
    function updateStepContent(step, data) {
        switch(step) {
            case 1:
                updateStep1(data);
                break;
            case 2:
                updateStep2(data);
                break;
            case 3:
                updateStep3(data);
                break;
            case 4:
                updateStep4(data);
                break;
        }
    }
    
    function updateStep1(data) {
        const textBreakdown = document.querySelector('#step-1 .text-breakdown');
        if (textBreakdown) {
            textBreakdown.innerHTML = `
                <div class="text-analysis">
                    <h4>Original Input Text:</h4>
                    <div class="original-text">${data.original_text || 'Text not available'}</div>
                    <div class="text-stats">
                        <p><strong>Character Count:</strong> ${data.original_text ? data.original_text.length : 0}</p>
                        <p><strong>Word Count:</strong> ${data.original_text ? data.original_text.split(' ').length : 0}</p>
                    </div>
                </div>
            `;
        }
    }
    
    function updateStep2(data) {
        const tokenBreakdown = document.querySelector('#step-2 .token-breakdown');
        if (tokenBreakdown && data.tokens) {
            const tokenHtml = data.tokens.map((token, index) => {
                const isSubword = token.startsWith('##');
                const isSpecial = token === '[CLS]' || token === '[SEP]';
                let tokenClass = 'token-item';
                if (isSubword) tokenClass += ' subword-token';
                else if (isSpecial) tokenClass += ' special-token';
                else tokenClass += ' full-token';
                
                return `
                    <div class="${tokenClass}">
                        <span class="token-text">${token}</span>
                        <span class="token-type">${isSubword ? 'Subword' : isSpecial ? 'Special' : 'Full Token'}</span>
                    </div>
                `;
            }).join('');
            
            tokenBreakdown.innerHTML = `
                <div class="tokenization-results">
                    <h4>Tokenization Results:</h4>
                    <div class="token-list-display">${tokenHtml}</div>
                    <div class="token-stats">
                        <p><strong>Total Tokens:</strong> ${data.tokens.length}</p>
                        <p><strong>Full Tokens:</strong> ${data.tokens.filter(t => !t.startsWith('##') && t !== '[CLS]' && t !== '[SEP]').length}</p>
                        <p><strong>Subword Tokens:</strong> ${data.tokens.filter(t => t.startsWith('##')).length}</p>
                    </div>
                </div>
            `;
        }
    }
    
    function updateStep3(data) {
        const tokenIdsBreakdown = document.querySelector('#step-3 .token-ids-breakdown');
        if (tokenIdsBreakdown && data.tokens && data.token_ids) {
            const tokenIdsHtml = data.tokens.map((token, index) => {
                const tokenId = data.token_ids[index];
                return `
                    <div class="token-id-item">
                        <span class="token-text">${token}</span>
                        <span class="token-id">ID: ${tokenId}</span>
                    </div>
                `;
            }).join('');
            
            tokenIdsBreakdown.innerHTML = `
                <div class="token-ids-results">
                    <h4>Token to ID Mapping:</h4>
                    <div class="token-ids-display">${tokenIdsHtml}</div>
                    <div class="token-ids-stats">
                        <p><strong>Vocabulary Size:</strong> ${data.technical_info ? data.technical_info.vocab_size.toLocaleString() : 'N/A'}</p>
                        <p><strong>ID Range:</strong> ${Math.min(...data.token_ids)} - ${Math.max(...data.token_ids)}</p>
                    </div>
                </div>
            `;
        }
    }
    
    function updateStep4(data) {
        const embeddingBreakdown = document.querySelector('#step-4 .embedding-breakdown');
        if (embeddingBreakdown && data.embeddings) {
            const sampleEmbeddings = data.embeddings.slice(0, 5); // Show first 5 tokens
            const embeddingHtml = sampleEmbeddings.map((embedding, index) => {
                const token = data.tokens[index];
                const vectorPreview = embedding.slice(0, 10).map(val => val.toFixed(3)).join(', ');
                return `
                    <div class="embedding-item">
                        <span class="token-text">${token}</span>
                        <span class="vector-preview">[${vectorPreview}...]</span>
                        <span class="vector-dim">${embedding.length} dimensions</span>
                    </div>
                `;
            }).join('');
            
            embeddingBreakdown.innerHTML = `
                <div class="embedding-results">
                    <h4>Vector Embeddings (Sample):</h4>
                    <div class="embedding-display">${embeddingHtml}</div>
                    <div class="embedding-stats">
                        <p><strong>Original Dimensions:</strong> ${data.technical_info ? data.technical_info.embedding_shape[1] : 'N/A'}</p>
                        <p><strong>Reduced to 3D:</strong> For visualization</p>
                        <p><strong>Total Embeddings:</strong> ${data.embeddings.length}</p>
                    </div>
                </div>
            `;
        }
    }
    
    processBtn.addEventListener('click', function() {
        processBtn.disabled = true;
        processBtn.textContent = 'Processing...';
        
        const text = inputText.value.trim();
        
        if (text.length === 0) {
            alert('Please enter some text');
            processBtn.disabled = false;
            processBtn.textContent = 'Process';
            return;
        }
        
        console.log('Processing text:', text);
        
        console.log('Sending request to /process with text:', text);
        
        fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                include_gemini: false
            }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Received data:', data);
            console.log('Plot data structure:', data.plot_data);
            console.log('Plot data keys:', Object.keys(data.plot_data || {}));
            console.log('Plot data.data:', data.plot_data?.data);
            console.log('Plot data.data[0]:', data.plot_data?.data?.[0]);
            
            // Store data for step-by-step process
            processData = {
                ...data,
                original_text: inputText.value.trim()
            };
            
            // Update all visualizations
            updateTokenList(data.tokens, data.token_ids);
            updateTechnicalInfo(data);  // Pass the entire data object
            updateEmbeddingPlot(data.plot_data);
            updateAttentionHeatmap(data.attention_scores, data.tokens);
            
            
            // Show analysis panels
            document.getElementById('clustering-analysis-panel').style.display = 'block';
            document.getElementById('attention-analysis-panel').style.display = 'block';
            
            // Initialize step-by-step process
            currentStep = 1;
            updateStepDisplay();
        })
        .catch(error => {
            console.error('Error details:', error);
            console.error('Error name:', error.name);
            console.error('Error message:', error.message);
            console.error('Error stack:', error.stack);
            
            // More specific error handling
            if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
                alert('Network error: Unable to connect to the server. Please check if the Flask application is running.');
            } else {
                alert(`Error: ${error.message}`);
            }
        })
        .finally(() => {
            processBtn.disabled = false;
            processBtn.textContent = 'Process';
        });
    });

    function updateTokenList(tokens, tokenIds) {
        const tokenList = document.getElementById('token-list');
        tokenList.innerHTML = '';
        tokens.forEach((token, index) => {
            const li = document.createElement('li');
            li.className = token.startsWith('##') ? 'subword-token' : 'full-token';
            const displayInfo = `
                <div class="token-content">
                    <span class="token-text">${token}</span>
                    <span class="token-id">ID: ${tokenIds[index]}</span>
                    ${token.startsWith('##') ? 
                        '<span class="token-type">Subword continuation</span>' : 
                        '<span class="token-type">Full/Start token</span>'}
                </div>
            `;
            li.innerHTML = displayInfo;
            tokenList.appendChild(li);
        });
    }

    function updateTechnicalInfo(data) {
        const info = data.technical_info;

        function createInfoRow(label, value) {
            return `
                <div class="info-detail">
                    <span class="info-label">${label}:</span>
                    <span class="info-value">${value}</span>
                </div>
            `;
        }

        // Model Details
        document.getElementById('model-details').innerHTML = `
            ${createInfoRow('Model', info.model_name)}
            ${createInfoRow('Vocabulary Size', info.vocab_size.toLocaleString())}
            ${createInfoRow('Hidden Layers', info.num_hidden_layers)}
            ${createInfoRow('Attention Heads', info.num_attention_heads)}
        `;

        // Input Analysis
        document.getElementById('input-details').innerHTML = `
            ${createInfoRow('Input Length', info.input_text_length)}
            ${createInfoRow('Tokens', info.num_tokens)}
            ${createInfoRow('Avg. Tokens/Word', (info.num_tokens / info.input_text_length).toFixed(2))}
        `;

        // Embedding Details
        document.getElementById('embedding-details').innerHTML = `
            ${createInfoRow('Original Dimensions', info.embedding_shape[1])}
            ${createInfoRow('Reduced Dimensions', '3D')}
            ${createInfoRow('Perplexity', info.perplexity)}
            ${createInfoRow('Hidden Size', info.hidden_size)}
        `;

        // Attention Details
        document.getElementById('attention-details').innerHTML = `
            ${createInfoRow('Attention Matrix', `${info.num_tokens} × ${info.num_tokens}`)}
            ${createInfoRow('Max Attention', Math.max(...data.attention_scores.flat()).toFixed(4))}
            ${createInfoRow('Min Attention', Math.min(...data.attention_scores.flat()).toFixed(4))}
            ${createInfoRow('Avg Attention', (data.attention_scores.flat().reduce((a, b) => a + b) / data.attention_scores.flat().length).toFixed(4))}
        `;
    }

    function updateEmbeddingPlot(plotData) {
        try {
            const container = document.getElementById('embedding-plot');
            
            // Clear any existing content
            container.innerHTML = '';
            
            // Fixed dimensions to match larger container
            const plotWidth = 800;
            const plotHeight = 650;
    
            // Validate plot data structure
            if (!plotData || !plotData.data || !Array.isArray(plotData.data) || plotData.data.length === 0) {
                console.error('Invalid plot data structure:', plotData);
                container.innerHTML = '<p style="color: red; text-align: center; padding: 20px;">Error: Invalid plot data structure</p>';
                return;
            }
            
            // Create a deep copy of the plot data to avoid modifying the original
            const data = JSON.parse(JSON.stringify(plotData.data));
            
            // Test with simple data if the original data seems problematic
            if (!data[0].x || !data[0].y || !data[0].z || data[0].x.length === 0) {
                console.warn('Data appears to be empty, creating test plot');
                data[0] = {
                    type: 'scatter3d',
                    x: [0, 1, -1, 0.5, -0.5],
                    y: [0, 0.5, -0.5, 1, -1],
                    z: [0, 0.3, -0.3, 0.7, -0.7],
                    mode: 'markers',
                    text: ['test1', 'test2', 'test3', 'test4', 'test5'],
                    marker: {
                        size: 8,
                        color: [0, 1, 2, 3, 4],
                        colorscale: 'Viridis',
                        opacity: 0.8
                    }
                };
            }
            
            // Ensure all required properties are present
            if (!data[0].type) data[0].type = 'scatter3d';
            if (!data[0].mode) data[0].mode = 'markers';
            if (!data[0].marker) data[0].marker = { size: 8, color: 'blue' };
            
            console.log('Creating 3D plot with data:', data[0]);
            
            // Configure marker with proper color mapping
            data[0].marker = {
                size: 6,
                opacity: 0.8,
                color: data[0].text.map((token, index) => index),
                colorscale: 'Viridis',
                line: {
                    color: 'rgba(0,0,0,0.3)',
                    width: 0.5
                },
                symbol: 'circle'
            };
            
            // Configure hover template
            data[0].hovertemplate = 
                '<b>Token:</b> %{text}<br>' +
                '<b>Position:</b> %{marker.color}<br>' +
                '<b>X:</b> %{x:.3f}<br>' +
                '<b>Y:</b> %{y:.3f}<br>' +
                '<b>Z:</b> %{z:.3f}<br>' +
                '<extra></extra>';
            
            // Handle text labels based on checkbox
            const showTextLabels = document.getElementById('show-text-labels').checked;
            if (showTextLabels) {
                data[0].mode = 'markers+text';
                    data[0].textposition = 'top center';
                    data[0].textfont = {
                    size: 8,
                    color: 'black'
                    };
            } else {
                data[0].mode = 'markers';
            }
    
            // Clean layout configuration with fixed dimensions
            const layout = {
                width: plotWidth,
                height: plotHeight,
                    title: {
                        text: '3D Token Embeddings',
                    font: { size: 14 }
                    },
                    scene: {
                        aspectmode: 'cube',
                        camera: {
                            up: {x: 0, y: 0, z: 1},
                            center: {x: 0, y: 0, z: 0},
                            eye: {x: 1.5, y: 1.5, z: 1.5}
                        },
                        xaxis: {
                            title: 'X',
                            showgrid: true,
                        zeroline: true,
                        gridcolor: 'rgba(0,0,0,0.1)'
                        },
                        yaxis: {
                            title: 'Y',
                            showgrid: true,
                        zeroline: true,
                        gridcolor: 'rgba(0,0,0,0.1)'
                        },
                        zaxis: {
                            title: 'Z',
                            showgrid: true,
                        zeroline: true,
                        gridcolor: 'rgba(0,0,0,0.1)'
                        }
                    },
                    margin: {
                    l: 40,
                    r: 40,
                    t: 40,
                    b: 40,
                    pad: 0
                },
                showlegend: false,
                coloraxis: {
                    colorbar: {
                        x: -0.05,
                        thickness: 12,
                        len: 0.7,
                        title: {
                            text: 'Token Index',
                            side: 'right',
                            font: { size: 10 }
                        }
                    }
                }
            };
    
            const config = {
                responsive: false,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'token_embeddings',
                    height: plotHeight,
                    width: plotWidth,
                    scale: 2
                }
            };
    
            console.log('Creating plot with fixed dimensions:', plotWidth, 'x', plotHeight);
            
            // Create the plot
                Plotly.newPlot('embedding-plot', data, layout, config).then(function() {
                console.log('Plot created successfully with fixed dimensions');
                }).catch(function(error) {
                    console.error('Error creating plot:', error);
                    // Fallback to simple test plot
                    console.log('Trying fallback test plot...');
                    const fallbackData = [{
                        type: 'scatter3d',
                        x: [0, 1, -1],
                        y: [0, 1, -1],
                        z: [0, 1, -1],
                        mode: 'markers',
                        marker: {
                        size: 8,
                            color: 'red'
                        }
                    }];
                    const fallbackLayout = {
                    width: plotWidth,
                    height: plotHeight,
                        scene: {
                            xaxis: {title: 'X'},
                            yaxis: {title: 'Y'},
                            zaxis: {title: 'Z'}
                        }
                    };
                    Plotly.newPlot('embedding-plot', fallbackData, fallbackLayout, config);
                });
            
            // Add event listener for point selection
            container.on('plotly_click', function(data) {
                const point = data.points[0];
                console.log('Clicked token:', point.text);
                console.log('Position:', point.marker.color);
                console.log('Coordinates:', [point.x, point.y, point.z]);
            });
    
        } catch (error) {
            console.error('Error creating 3D plot:', error);
            console.error('Plot data:', plotData);
        }
    }
    
    function createColorScaleSelector(colorscales) {
        // Remove existing selector if it exists
        const existingSelector = document.querySelector('.color-scale-selector');
        if (existingSelector) {
            existingSelector.remove();
        }
    
        // Find the controls container
        const container = document.querySelector('.color-scale-selector');
        if (!container) return;
        
        // Create color scale selector
        container.innerHTML = `
            <label for="colorscale">Color Scale:</label>
            <select id="colorscale">
                ${colorscales.map(scale => `
                    <option value="${scale}" ${scale === 'Viridis' ? 'selected' : ''}>${scale}</option>
                `).join('')}
            </select>
        `;
    
        // Add event listener
        document.getElementById('colorscale').addEventListener('change', function(e) {
            Plotly.restyle('embedding-plot', {
                'marker.colorscale': e.target.value
            });
        });
    }
    
    // Add CSS
    const style = document.createElement('style');
    style.textContent = `
        .viz-container {
            margin: 20px 0;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 100%;
            box-sizing: border-box;
        }
    
        #embedding-plot {
            width: 100%;
            min-height: 400px;
            background-color: white;
        }
    
        .color-scale-selector {
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
    
        .color-scale-selector label {
            margin-right: 10px;
            font-weight: 500;
        }
    
        .color-scale-selector select {
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
    
        /* Add responsive design */
        @media (max-width: 768px) {
            .viz-container {
                padding: 10px;
            }
            
            #embedding-plot {
                min-height: 300px;
            }
        }
    `;
    document.head.appendChild(style);
    
    
    

    function updateAttentionHeatmap(attentionScores, tokens) {
        // Fixed dimensions to match larger container
        const heatmapWidth = 800;
        const heatmapHeight = 650;
        const fontSize = Math.max(8, Math.min(12, 200 / tokens.length));
        
        // Improve color scale for better attention visualization
        const data = [{
            z: attentionScores,
            x: tokens,
            y: tokens,
            type: 'heatmap',
            colorscale: [
                [0, 'rgb(0,0,100)'],      // Dark blue for low attention
                [0.2, 'rgb(0,100,200)'],   // Blue for moderate attention
                [0.4, 'rgb(0,200,255)'],   // Light blue for higher attention
                [0.6, 'rgb(100,255,100)'], // Green for high attention
                [0.8, 'rgb(255,255,0)'],   // Yellow for very high attention
                [1, 'rgb(255,200,0)']      // Orange for highest attention
            ],
            hoverongaps: false,
            hoverinfo: 'all',
            hovertemplate: 
                'Source: %{y}<br>' +
                'Target: %{x}<br>' +
                'Attention: %{z:.4f}<br>' +
                '<extra></extra>',
        }];
    
        const layout = {
            title: {
                text: 'Attention Scores<br><sub>Hover over cells to see attention values</sub>',
                font: { size: 16 },
                y: 0.95
            },
            xaxis: {
                title: 'Target Tokens',
                side: 'bottom',
                tickfont: { size: fontSize },
                tickangle: -45
            },
            yaxis: {
                title: 'Source Tokens',
                tickfont: { size: fontSize }
            },
            width: heatmapWidth,
            height: heatmapHeight,
            margin: {
                l: 60,  // Left margin for y-axis labels
                r: 20,  // Right margin
                t: 50,  // Top margin
                b: 60   // Bottom margin for x-axis labels
            },
            annotations: [],
            coloraxis: {
                colorbar: {
                    title: 'Attention Score',
                    titleside: 'right',
                    thickness: 8,
                    len: 0.5,
                    x: -0.15  // Move colorbar to the left
                }
            }
        };
    
        // Add annotations for strong attention values with improved positioning
        const showAttentionValues = document.getElementById('show-attention-values')?.checked ?? true;
        
        if (showAttentionValues) {
            const threshold = 0.1; // Lower threshold to show more attention patterns
            const maxScore = Math.max(...attentionScores.flat());
            const minScore = Math.min(...attentionScores.flat());
            
            attentionScores.forEach((row, i) => {
                row.forEach((score, j) => {
                    // Show self-attention (diagonal) and strong cross-attention
                    if (score > threshold || (i === j && score > maxScore * 0.3)) {
                        layout.annotations.push({
                            x: j,
                            y: i,
                            text: score.toFixed(3),
                            showarrow: false,
                            xanchor: 'center',
                            yanchor: 'middle',
                            font: {
                                color: score > maxScore * 0.6 ? 'white' : 'black',
                                size: Math.max(6, Math.min(fontSize, 10)) // Limit font size
                            }
                        });
                    }
                });
            });
        }
    
        Plotly.newPlot('attention-heatmap', data, layout, {
            responsive: false,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            toImageButtonOptions: {
                format: 'png',
                filename: 'attention_heatmap',
                height: heatmapHeight,
                width: heatmapWidth,
                scale: 2
            }
        });
    }
    

    function showSampleGeneratorModal() {
        // Create a modal for sample generation form
        const modal = document.createElement('div');
        modal.className = 'sample-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>🎲 Generate Sample Text</h3>
                    <button class="close-modal">&times;</button>
                </div>
                <div class="modal-body">
                    <form id="sample-generator-form">
                        <div class="form-group">
                            <label for="sample-topic">Topic:</label>
                            <input type="text" id="sample-topic" placeholder="e.g., Artificial Intelligence, Cooking, Space Exploration" value="Artificial Intelligence">
                        </div>
                        <div class="form-group">
                            <label for="sample-complexity">Complexity:</label>
                            <select id="sample-complexity">
                                <option value="simple">Simple</option>
                                <option value="medium" selected>Medium</option>
                                <option value="complex">Complex</option>
                            </select>
                        </div>
                        <div class="form-actions">
                            <button type="button" class="cancel-btn" onclick="this.closest('.sample-modal').remove()">Cancel</button>
                            <button type="submit" class="generate-btn">Generate</button>
                        </div>
                    </form>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Add event listeners
        modal.querySelector('.close-modal').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        modal.querySelector('#sample-generator-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const topic = document.getElementById('sample-topic').value.trim();
            const complexity = document.getElementById('sample-complexity').value;
            
            if (!topic) {
                alert('Please enter a topic');
                return;
            }
            
            generateAndUseSample(topic, complexity, modal);
        });
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });
    }
    
    function generateAndUseSample(topic, complexity, modal) {
        const generateBtn = modal.querySelector('.generate-btn');
        const originalText = generateBtn.textContent;
        
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
        
        fetch('/generate-sample', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                topic: topic,
                complexity: complexity
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.sample_text) {
                // Insert the generated text into the input field
                document.getElementById('input-text').value = data.sample_text;
                // Close the modal
                document.body.removeChild(modal);
            } else {
                alert('Error generating sample: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error generating sample: ' + error.message);
        })
        .finally(() => {
            generateBtn.disabled = false;
            generateBtn.textContent = originalText;
        });
    }
    
    function updateClusteringAnalysis(analysis) {
        const content = document.getElementById('clustering-analysis-content');
        
        // Parse the analysis text more carefully
        // Look for actual cluster sections, not individual markdown elements
        let clusterSections = [];
        
        // Try to split by actual cluster boundaries
        // Look for patterns like "Cluster X:" or major section breaks
        if (analysis.includes('Cluster ')) {
            // Split by "Cluster \d+:" but keep the cluster number
            clusterSections = analysis.split(/(?=Cluster \d+:)/);
        } else {
            // If no explicit cluster numbers, try to find major sections
            // Look for double newlines or major markdown headers
            const sections = analysis.split(/\n\s*\n/);
            clusterSections = sections.filter(section => 
                section.trim().length > 50 && // Must be substantial content
                (section.includes('Tokens:') || section.includes('Theme:') || section.includes('Position:'))
            );
        }
        
        // If we still don't have good sections, treat the whole analysis as one section
        if (clusterSections.length === 0) {
            clusterSections = [analysis];
        }
        
        let formattedHTML = '';
        clusterSections.forEach((section, index) => {
            if (section.trim()) {
                formattedHTML += `<div class="pattern-section">${section.trim()}</div>`;
            }
        });
        
        content.innerHTML = formattedHTML;
        
        // Apply formatting to each cluster section
        const sections = content.querySelectorAll('.pattern-section');
        sections.forEach((section, index) => {
            // Store the original text for token extraction
            section.dataset.originalText = section.textContent;
            formatClusterContent(section);
        });
    }
    
    function formatClusterContent(section) {
        const content = section.innerHTML;
        
        // Parse the cluster information more intelligently
        let formattedContent = content;
        
        // Extract cluster number and create a header - try multiple patterns
        let clusterMatch = content.match(/Cluster (\d+):/);
        let clusterNumber = '1';
        
        if (clusterMatch) {
            clusterNumber = clusterMatch[1];
        } else {
            // If no cluster number found, try to extract from the section index
            const allSections = document.querySelectorAll('.pattern-section');
            const currentIndex = Array.from(allSections).indexOf(section);
            clusterNumber = (currentIndex + 1).toString();
        }
        
        // Define cluster colors
        const clusterColors = [
            '#e74c3c', // Red
            '#3498db', // Blue  
            '#2ecc71', // Green
            '#f39c12', // Orange
            '#9b59b6', // Purple
            '#1abc9c', // Turquoise
            '#e67e22', // Carrot
            '#34495e'  // Dark blue
        ];
        
        const clusterColor = clusterColors[(parseInt(clusterNumber) - 1) % clusterColors.length];
        
        // Extract and format the cluster title (handle markdown **text**)
        const titleMatch = content.match(/\*\*([^*]+)\*\*/);
        const clusterTitle = titleMatch ? titleMatch[1] : `Cluster ${clusterNumber}`;
        
        // Replace the cluster header with styled version - handle multiple patterns more carefully
        // Pattern 1: Cluster X: **Title**
        formattedContent = formattedContent.replace(/Cluster \d+:\s*\*\*([^*]+)\*\*/, `<div class="cluster-header" style="color: #2c3e50; font-weight: bold; margin-bottom: 8px;">$1</div>`);
        
        // Pattern 2: Just **Title** (standalone) - but be more selective
        // Only replace if it's at the beginning of the content and looks like a title
        formattedContent = formattedContent.replace(/^\s*\*\*([^*]+)\*\*/, `<div class="cluster-header" style="color: #2c3e50; font-weight: bold; margin-bottom: 8px;">$1</div>`);
        
        // Replace **Tokens:** with proper formatting and ensure tokens are properly quoted
        formattedContent = formattedContent.replace(/\*\*Tokens:\*\*/g, '<div class="pattern-label">Tokens:</div><div class="pattern-tokens">');
        
        // Replace **Theme:** with proper formatting
        formattedContent = formattedContent.replace(/\*\*Theme:\*\*/g, '</div><div class="pattern-label">Theme:</div><div class="pattern-content">');
        
        // Replace **Position:** with proper formatting
        formattedContent = formattedContent.replace(/\*\*Position:\*\*/g, '</div><div class="pattern-label">Position:</div><div class="pattern-content">');
        
        // Close the last content div
        formattedContent += '</div>';
        
        // Add clickable cluster section for highlighting with visual indicator
        formattedContent = `<div class="cluster-clickable" data-cluster="${clusterNumber}" data-color="${clusterColor}" onclick="console.log('Cluster clicked:', ${clusterNumber}); highlightClusterInPlot(${clusterNumber}, '${clusterColor}')" title="Click to highlight this cluster in the 3D plot">${formattedContent}<div class="highlight-hint">🔍 Click to highlight in plot</div></div>`;
        
        section.innerHTML = formattedContent;
        
        // Store the original text content for token extraction
        section.dataset.originalText = section.textContent;
    }
    
    function highlightClusterTokens(section, clusterIndex) {
        if (!processData || !processData.tokens) return;
        
        // Extract tokens from the section text
        const sectionText = section.textContent;
        const tokens = [];
        
        // Find tokens mentioned in the section (look for quoted tokens)
        const tokenMatches = sectionText.match(/'([^']+)'/g);
        if (tokenMatches) {
            tokenMatches.forEach(match => {
                const token = match.replace(/'/g, '');
                if (processData.tokens.includes(token)) {
                    tokens.push(token);
                }
            });
        }
        
        // Highlight tokens in the 3D plot
        if (tokens.length > 0 && window.Plotly) {
            const plotElement = document.getElementById('embedding-plot');
            if (plotElement && plotElement.data) {
                // Create highlight effect by updating marker colors
                const update = {
                    'marker.color': plotElement.data[0].text.map(token => 
                        tokens.includes(token) ? '#ff6b6b' : '#95a5a6'
                    ),
                    'marker.size': plotElement.data[0].text.map(token => 
                        tokens.includes(token) ? 12 : 8
                    )
                };
                
                Plotly.restyle('embedding-plot', update);
            }
        }
    }
    
    // Global function for highlighting clusters in the plot
    window.highlightClusterInPlot = function(clusterNumber, clusterColor) {
        console.log('=== highlightClusterInPlot START ===');
        console.log('Function called with clusterNumber:', clusterNumber, 'clusterColor:', clusterColor);
        console.log('processData available:', !!processData);
        console.log('processData.tokens available:', !!(processData && processData.tokens));
        
        try {
            if (!processData || !processData.tokens) {
                console.log('No processData or tokens available');
                return;
            }
        
        // Find the cluster section and extract tokens
        const clusterSections = document.querySelectorAll('.pattern-section');
        console.log('Found cluster sections:', clusterSections.length);
        let targetSection = null;
        
        clusterSections.forEach((section, index) => {
            console.log(`Section ${index}:`, section.textContent.substring(0, 100));
            // Try multiple patterns to find the cluster section
            if (section.textContent.includes(`Cluster ${clusterNumber}:`) || 
                section.dataset.cluster === clusterNumber.toString() ||
                section.querySelector(`[data-cluster="${clusterNumber}"]`)) {
                targetSection = section;
                console.log('Found target section for cluster', clusterNumber);
            }
        });
        
        if (!targetSection) return;
        
        // Use the original text content for better token extraction
        const sectionText = targetSection.dataset.originalText || targetSection.textContent;
        const tokens = [];
        
        console.log('Section text for token extraction:', sectionText);
        
        // Extract tokens from the cluster description more precisely
        // Look for the Tokens: line specifically
        const lines = sectionText.split('\n');
        let tokensLine = null;
        
        for (let line of lines) {
            if (line.includes('Tokens:')) {
                tokensLine = line;
                break;
            }
        }
        
        // If no tokens line found in text, try to extract from the formatted content
        if (!tokensLine) {
            const tokensElement = targetSection.querySelector('.pattern-tokens');
            if (tokensElement) {
                tokensLine = tokensElement.textContent;
                console.log('Found tokens from formatted element:', tokensLine);
            }
        }
        
        if (tokensLine) {
            // Extract tokens after "Tokens:" - handle both quoted and unquoted tokens
            const tokenMatches = tokensLine.match(/Tokens:\s*(.+)/);
            if (tokenMatches) {
                const tokenString = tokenMatches[1];
                console.log('Token string to parse:', tokenString);
                
                // Remove quotes and split by comma
                const cleanTokenString = tokenString.replace(/['"]/g, '');
                const tokenCandidates = cleanTokenString.split(',').map(t => t.trim()).filter(t => t.length > 0);
                
                console.log('Token candidates:', tokenCandidates);
                
                // Match against actual tokens in the data
                tokenCandidates.forEach(candidate => {
                    // Try exact match first
                    if (processData.tokens.includes(candidate)) {
                        tokens.push(candidate);
                    } else {
                        // Try case-insensitive match
                        const lowerCandidate = candidate.toLowerCase();
                        const matchingToken = processData.tokens.find(token => token.toLowerCase() === lowerCandidate);
                        if (matchingToken && !tokens.includes(matchingToken)) {
                            tokens.push(matchingToken);
                        }
                    }
                });
            }
        }
        
        console.log(`Cluster ${clusterNumber} tokens:`, tokens);
        
        // Highlight tokens in the 3D plot
        if (tokens.length > 0 && window.Plotly) {
            const plotElement = document.getElementById('embedding-plot');
            if (plotElement && plotElement.data) {
                console.log('Highlighting tokens:', tokens);
                console.log('Cluster color:', clusterColor);
                
                // Get the current plot data
                const currentData = plotElement.data[0];
                const allTokens = currentData.text;
                
                // Create new color array - use cluster color for highlighted tokens, grey for others
                const newColors = allTokens.map((token, index) => {
                    if (tokens.includes(token)) {
                        return clusterColor; // Use the cluster color for highlighted tokens
                    } else {
                        return '#bdc3c7'; // Grey for non-highlighted tokens
                    }
                });
                
                // Create new size array - larger for highlighted tokens
                const newSizes = allTokens.map((token, index) => {
                    if (tokens.includes(token)) {
                        return 12; // Larger size for highlighted tokens
                    } else {
                        return 6; // Normal size for others
                    }
                });
                
                console.log('New colors:', newColors);
                console.log('New sizes:', newSizes);
                
                // Update the plot with new colors and sizes
                Plotly.restyle('embedding-plot', {
                    'marker.color': [newColors],
                    'marker.size': [newSizes]
                });
                
            }
        }
        
        } catch (error) {
            console.error('Error in highlightClusterInPlot:', error);
        }
        
        console.log('=== highlightClusterInPlot END ===');
    }
    
    // Fallback function to ensure highlighting is available
    window.testHighlight = function() {
        console.log('Test highlight function called');
    }
    
    
    // Clear highlight functions removed - no longer needed
    
    // Fullscreen functionality - make globally accessible
    window.openFullscreen = function(plotId) {
        const modal = document.getElementById('fullscreen-modal');
        const title = document.getElementById('fullscreen-title');
        const container = document.getElementById('fullscreen-plot-container');
        const originalPlot = document.getElementById(plotId);
        
        // Set the title based on plot type
        if (plotId === 'embedding-plot') {
            title.textContent = '3D Token Embeddings - Fullscreen';
        } else if (plotId === 'attention-heatmap') {
            title.textContent = 'Attention Visualization - Fullscreen';
        }
        
        // Clone the plot container
        container.innerHTML = '';
        const plotClone = originalPlot.cloneNode(true);
        plotClone.id = plotId + '-fullscreen';
        container.appendChild(plotClone);
        
        // Show the modal
        modal.style.display = 'flex';
        
        // Recreate the plot in fullscreen with larger dimensions
        setTimeout(() => {
            console.log('Fullscreen data check:', { plotId, processData });
            if (plotId === 'embedding-plot' && processData) {
                // Try different possible data structures
                const embeddings_3d = processData.embeddings_3d || processData.embeddings || processData.embeddings_3d_data;
                const tokens = processData.tokens || processData.token_list;
                
                if (embeddings_3d && tokens) {
                    console.log('Using embeddings_3d:', embeddings_3d);
                    updateEmbeddingPlotFullscreen(embeddings_3d, tokens);
                } else {
                    console.log('No valid embedding data found, trying fallback');
                    // Fallback: clone existing plot
                    const existingPlot = document.getElementById(plotId);
                    if (existingPlot && existingPlot.innerHTML.trim()) {
                        const fullscreenContainer = document.getElementById(plotId + '-fullscreen');
                        fullscreenContainer.innerHTML = existingPlot.innerHTML;
                        
                        // Try to resize the plot
                        if (window.Plotly && fullscreenContainer.querySelector('.plotly')) {
                            const plotWidth = window.innerWidth * 0.9;
                            const plotHeight = window.innerHeight * 0.8;
                            Plotly.relayout(plotId + '-fullscreen', {
                                width: plotWidth,
                                height: plotHeight
                            });
                        }
                    }
                }
            } else if (plotId === 'attention-heatmap' && processData && processData.attention_scores) {
                updateAttentionHeatmapFullscreen(processData.attention_scores, processData.tokens);
            } else {
                console.log('No processData available, using fallback');
                // If no processData, try to clone the existing plot
                const existingPlot = document.getElementById(plotId);
                if (existingPlot && existingPlot.innerHTML.trim()) {
                    const fullscreenContainer = document.getElementById(plotId + '-fullscreen');
                    fullscreenContainer.innerHTML = existingPlot.innerHTML;
                    
                    // Try to resize the plot
                    if (window.Plotly && fullscreenContainer.querySelector('.plotly')) {
                        const plotWidth = window.innerWidth * 0.9;
                        const plotHeight = window.innerHeight * 0.8;
                        Plotly.relayout(plotId + '-fullscreen', {
                            width: plotWidth,
                            height: plotHeight
                        });
                    }
                }
            }
        }, 100);
    }
    
    window.closeFullscreen = function() {
        const modal = document.getElementById('fullscreen-modal');
        modal.style.display = 'none';
    }
    
    // Close modal when clicking outside
    document.addEventListener('click', function(event) {
        const modal = document.getElementById('fullscreen-modal');
        if (event.target === modal) {
            closeFullscreen();
        }
    });
    
    // Close modal with Escape key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeFullscreen();
        }
    });
    
    window.updateEmbeddingPlotFullscreen = function(embeddings_3d, tokens) {
        try {
            console.log('Creating fullscreen 3D plot with data:', embeddings_3d, tokens);
            const container = document.getElementById('embedding-plot-fullscreen');
            container.innerHTML = '';
            
            // Use larger dimensions for fullscreen
            const plotWidth = window.innerWidth * 0.9;
            const plotHeight = window.innerHeight * 0.8;
            
            if (!embeddings_3d || !Array.isArray(embeddings_3d) || embeddings_3d.length === 0) {
                console.error('Invalid 3D embedding data structure:', embeddings_3d);
                container.innerHTML = '<p style="color: red; text-align: center; padding: 20px;">Error: Invalid 3D embedding data structure</p>';
                return;
            }
            
            // Use the already processed 3D embeddings from the backend
            const plotData = embeddings_3d.map(point => ({
                x: point[0],
                y: point[1], 
                z: point[2]
            }));
            
            const data = [{
                type: 'scatter3d',
                x: plotData.map(point => point.x),
                y: plotData.map(point => point.y),
                z: plotData.map(point => point.z),
                mode: 'markers',
                text: tokens,
                marker: {
                    size: 8,
                    color: tokens.map((token, index) => index),
                    colorscale: 'Viridis',
                    opacity: 0.8,
                    line: {
                        color: 'rgba(0,0,0,0.3)',
                        width: 0.5
                    },
                    symbol: 'circle'
                },
                hovertemplate: '<b>Token:</b> %{text}<br>' +
                    '<b>Position:</b> %{marker.color}<br>' +
                    '<b>X:</b> %{x:.3f}<br>' +
                    '<b>Y:</b> %{y:.3f}<br>' +
                    '<b>Z:</b> %{z:.3f}<br>' +
                    '<extra></extra>'
            }];
            
            const layout = {
                width: plotWidth,
                height: plotHeight,
                title: {
                    text: '3D Token Embeddings - Fullscreen View',
                    font: { size: 18 }
                },
                scene: {
                    aspectmode: 'cube',
                    camera: {
                        up: {x: 0, y: 0, z: 1},
                        center: {x: 0, y: 0, z: 0},
                        eye: {x: 1.5, y: 1.5, z: 1.5}
                    },
                    xaxis: {
                        title: 'X',
                        showgrid: true,
                        zeroline: true,
                        gridcolor: 'rgba(0,0,0,0.1)'
                    },
                    yaxis: {
                        title: 'Y',
                        showgrid: true,
                        zeroline: true,
                        gridcolor: 'rgba(0,0,0,0.1)'
                    },
                    zaxis: {
                        title: 'Z',
                        showgrid: true,
                        zeroline: true,
                        gridcolor: 'rgba(0,0,0,0.1)'
                    }
                },
                margin: { l: 60, r: 60, t: 60, b: 60, pad: 0 },
                showlegend: false,
                coloraxis: {
                    colorbar: {
                        x: -0.05,
                        thickness: 15,
                        len: 0.7,
                        title: {
                            text: 'Token Index',
                            side: 'right',
                            font: { size: 12 }
                        }
                    }
                }
            };
            
            const config = {
                responsive: false,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'token_embeddings_fullscreen',
                    height: plotHeight,
                    width: plotWidth,
                    scale: 2
                }
            };
            
            Plotly.newPlot('embedding-plot-fullscreen', data, layout, config);
            
        } catch (error) {
            console.error('Error creating fullscreen 3D plot:', error);
        }
    }
    
    window.updateAttentionHeatmapFullscreen = function(attentionScores, tokens) {
        try {
            const container = document.getElementById('attention-heatmap-fullscreen');
            container.innerHTML = '';
            
            // Use larger dimensions for fullscreen
            const heatmapWidth = window.innerWidth * 0.9;
            const heatmapHeight = window.innerHeight * 0.8;
            const fontSize = Math.max(10, Math.min(16, 300 / tokens.length));
            
            const data = [{
                z: attentionScores,
                x: tokens,
                y: tokens,
                type: 'heatmap',
                colorscale: [
                    [0, 'rgb(0,0,100)'],
                    [0.2, 'rgb(0,100,200)'],
                    [0.4, 'rgb(0,200,255)'],
                    [0.6, 'rgb(100,255,100)'],
                    [0.8, 'rgb(255,255,0)'],
                    [1, 'rgb(255,200,0)']
                ],
                hoverongaps: false,
                hoverinfo: 'all',
                hovertemplate: 
                    'Source: %{y}<br>' +
                    'Target: %{x}<br>' +
                    'Attention: %{z:.4f}<br>' +
                    '<extra></extra>',
            }];
            
            const layout = {
                title: {
                    text: 'Attention Scores - Fullscreen View<br><sub>Hover over cells to see attention values</sub>',
                    font: { size: 20 },
                    y: 0.95
                },
                xaxis: {
                    title: 'Target Tokens',
                    side: 'bottom',
                    tickfont: { size: fontSize },
                    tickangle: -45
                },
                yaxis: {
                    title: 'Source Tokens',
                    tickfont: { size: fontSize }
                },
                width: heatmapWidth,
                height: heatmapHeight,
                margin: {
                    l: 80,
                    r: 40,
                    t: 80,
                    b: 80
                },
                annotations: [],
                coloraxis: {
                    colorbar: {
                        title: 'Attention Score',
                        titleside: 'right',
                        thickness: 20,
                        len: 0.6,
                        x: -0.15
                    }
                }
            };
            
            // Add annotations for strong attention values
            const showAttentionValues = document.getElementById('show-attention-values')?.checked ?? true;
            if (showAttentionValues) {
                const threshold = 0.1;
                const maxScore = Math.max(...attentionScores.flat());
                
                attentionScores.forEach((row, i) => {
                    row.forEach((score, j) => {
                        if (score > threshold || (i === j && score > maxScore * 0.3)) {
                            layout.annotations.push({
                                x: j,
                                y: i,
                                text: score.toFixed(3),
                                showarrow: false,
                                xanchor: 'center',
                                yanchor: 'middle',
                                font: {
                                    color: score > maxScore * 0.6 ? 'white' : 'black',
                                    size: Math.max(8, Math.min(fontSize, 14))
                                }
                            });
                        }
                    });
                });
            }
            
            const config = {
                responsive: false,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'attention_heatmap_fullscreen',
                    height: heatmapHeight,
                    width: heatmapWidth,
                    scale: 2
                }
            };
            
            Plotly.newPlot('attention-heatmap-fullscreen', data, layout, config);
            
        } catch (error) {
            console.error('Error creating fullscreen attention heatmap:', error);
        }
    }
    
    function updateAttentionAnalysis(analysis) {
        const content = document.getElementById('attention-analysis-content');
        
        console.log('Raw attention analysis text:', analysis);
        
        // Try to parse the analysis text more intelligently
        let formattedContent = '';
        
        // Split the analysis into sections based on ** headers
        const sections = analysis.split(/(\*\*[^*]+\*\*:)/);
        
        console.log('Sections found:', sections.length);
        
        let currentSection = '';
        let currentTitle = '';
        
        for (let i = 0; i < sections.length; i++) {
            const section = sections[i].trim();
            
            if (section.match(/^\*\*[^*]+\*\*:$/)) {
                // This is a header
                if (currentSection && currentTitle) {
                    formattedContent += formatAttentionPatternContent(currentTitle, currentSection, formattedContent.split('pattern-section').length - 1);
                }
                currentTitle = section.replace(/^\*\*|\*\*:$/g, '');
                currentSection = '';
            } else if (section.length > 0) {
                // This is content
                currentSection += section + '\n';
            }
        }
        
        // Handle the last section
        if (currentSection && currentTitle) {
            formattedContent += formatAttentionPatternContent(currentTitle, currentSection, formattedContent.split('pattern-section').length - 1);
        }
        
        // If no sections were found, try to create a simple display
        if (!formattedContent) {
            formattedContent = createSimpleAttentionDisplay(analysis);
        }
        
        content.innerHTML = formattedContent;
    }
    
    function createSimpleAttentionDisplay(analysis) {
        // Create a simple display for the analysis text with markdown rendering
        return `
            <div class="pattern-section">
                <div class="pattern-header" style="color: #2c3e50; font-weight: bold; font-size: 14px; margin-bottom: 8px;">
                    Attention Analysis
                </div>
                <div class="pattern-content" style="color: #2c3e50; font-size: 12px; line-height: 1.4;">
                    ${renderMarkdown(analysis)}
                </div>
            </div>
        `;
    }
    
    function renderMarkdown(text) {
        if (!text) return '';
        
        // Convert markdown to HTML
        return text
            // Convert **text** to <strong>text</strong>
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Convert *text* to <em>text</em>
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Convert line breaks
            .replace(/\n/g, '<br>')
            // Convert bullet points
            .replace(/^- (.+)$/gm, '• $1');
    }
    
    function formatAttentionPatternContent(title, content, patternIndex) {
        console.log('Formatting section - Title:', title, 'Content:', content);
        
        const patternName = title || `Attention Pattern ${patternIndex + 1}`;
        
        // For sections that don't have the standard format, just display the content
        if (!content.includes('**Tokens:**') && !content.includes('Tokens:')) {
            return `
                <div class="pattern-section">
                    <div class="pattern-header" style="color: #2c3e50; font-weight: bold; font-size: 14px; margin-bottom: 8px;">
                        ${renderMarkdown(patternName)}
                    </div>
                    <div class="pattern-content" style="color: #2c3e50; font-size: 12px; line-height: 1.4;">
                        ${renderMarkdown(content)}
                    </div>
                </div>
            `;
        }
        
        // Extract tokens, theme, and position with multiple patterns
        let tokens = 'Not specified';
        let theme = 'Not specified';
        let position = 'Not specified';
        
        // Try multiple patterns for tokens
        const tokensPatterns = [
            /\*\*Tokens:\*\*\s*(.+?)(?=\*\*|$)/s,
            /Tokens:\s*(.+?)(?=\n|$)/s,
            /- Tokens:\s*(.+?)(?=\n|$)/s
        ];
        
        for (const pattern of tokensPatterns) {
            const match = content.match(pattern);
            if (match) {
                tokens = match[1].trim();
                break;
            }
        }
        
        // Try multiple patterns for theme
        const themePatterns = [
            /\*\*Theme:\*\*\s*(.+?)(?=\*\*|$)/s,
            /Theme:\s*(.+?)(?=\n|$)/s,
            /- Theme:\s*(.+?)(?=\n|$)/s
        ];
        
        for (const pattern of themePatterns) {
            const match = content.match(pattern);
            if (match) {
                theme = match[1].trim();
                break;
            }
        }
        
        // Try multiple patterns for position
        const positionPatterns = [
            /\*\*Position:\*\*\s*(.+?)(?=\*\*|$)/s,
            /Position:\s*(.+?)(?=\n|$)/s,
            /- Position:\s*(.+?)(?=\n|$)/s
        ];
        
        for (const pattern of positionPatterns) {
            const match = content.match(pattern);
            if (match) {
                position = match[1].trim();
                break;
            }
        }
        
        console.log('Extracted values:', { tokens, theme, position });
        
        // Define colors for different pattern types
        const patternColors = [
            '#e74c3c', // Red
            '#3498db', // Blue  
            '#2ecc71', // Green
            '#f39c12', // Orange
            '#9b59b6', // Purple
            '#1abc9c', // Turquoise
            '#e67e22', // Carrot
            '#34495e'  // Dark blue-grey
        ];
        
        const patternColor = patternColors[patternIndex % patternColors.length];
        
        // Format the content with proper styling and make it clickable
        const formattedContent = `
            <div class="pattern-section pattern-clickable" data-pattern-index="${patternIndex}" data-color="${patternColor}" onclick="highlightAttentionPatternInPlot(${patternIndex}, '${patternColor}')" title="Click to highlight this attention pattern in the heatmap">
                <div class="pattern-header" style="color: ${patternColor}; font-weight: bold; font-size: 14px; margin-bottom: 8px;">
                    ${renderMarkdown(patternName)}
                </div>
                <div class="pattern-content">
                    <div class="pattern-tokens" style="color: #2c3e50; font-size: 12px; margin-bottom: 4px;">
                        <strong>Tokens:</strong> ${renderMarkdown(tokens)}
                    </div>
                    <div class="pattern-theme" style="color: #2c3e50; font-size: 12px; margin-bottom: 4px;">
                        <strong>Theme:</strong> ${renderMarkdown(theme)}
                    </div>
                    <div class="pattern-position" style="color: #2c3e50; font-size: 12px; margin-bottom: 8px;">
                        <strong>Position:</strong> ${renderMarkdown(position)}
                    </div>
                </div>
                <div class="highlight-hint" style="color: #7f8c8d; font-size: 11px; font-style: italic; text-align: center; margin-top: 5px;">
                    🔍 Click to highlight in plot
                </div>
            </div>
        `;
        
        return formattedContent;
    }
    
    // Global function for highlighting attention patterns in the heatmap
    window.highlightAttentionPatternInPlot = function(patternIndex, patternColor) {
        console.log('highlightAttentionPatternInPlot called with:', patternIndex, patternColor);
        if (!processData || !processData.tokens) {
            console.log('No processData or tokens available');
            return;
        }
        
        // Find the pattern section and extract tokens
        const patternSections = document.querySelectorAll('.pattern-section');
        console.log('Found pattern sections:', patternSections.length);
        let targetSection = null;
        
        patternSections.forEach((section, index) => {
            console.log(`Section ${index}:`, section.textContent.substring(0, 100));
            if (section.dataset.patternIndex == patternIndex) {
                targetSection = section;
                console.log('Found target section for pattern', patternIndex);
            }
        });
        
        if (!targetSection) return;
        
        // Extract tokens from the pattern description
        const sectionText = targetSection.textContent;
        const tokens = [];
        
        // Look for the Tokens: line specifically
        const lines = sectionText.split('\n');
        let tokensLine = null;
        
        for (let line of lines) {
            if (line.includes('Tokens:')) {
                tokensLine = line;
                break;
            }
        }
        
        if (tokensLine) {
            // Extract tokens after "Tokens:" - handle both quoted and unquoted tokens
            const tokenMatches = tokensLine.match(/Tokens:\s*(.+)/);
            if (tokenMatches) {
                const tokenString = tokenMatches[1];
                console.log('Token string to parse:', tokenString);
                
                // Remove quotes and split by comma
                const cleanTokenString = tokenString.replace(/['"]/g, '');
                const tokenCandidates = cleanTokenString.split(',').map(t => t.trim()).filter(t => t.length > 0);
                
                console.log('Token candidates:', tokenCandidates);
                
                // Match against actual tokens in the data
                tokenCandidates.forEach(candidate => {
                    // Try exact match first
                    if (processData.tokens.includes(candidate)) {
                        tokens.push(candidate);
                    } else {
                        // Try case-insensitive match
                        const lowerCandidate = candidate.toLowerCase();
                        const matchingToken = processData.tokens.find(token => 
                            token.toLowerCase() === lowerCandidate
                        );
                        if (matchingToken) {
                            tokens.push(matchingToken);
                        }
                    }
                });
            }
        }
        
        console.log(`Pattern ${patternIndex} tokens:`, tokens);
        
        // Highlight tokens in the attention heatmap
        if (tokens.length > 0 && window.Plotly) {
            const plotElement = document.getElementById('attention-heatmap');
            if (plotElement && plotElement.data) {
                console.log('Highlighting attention pattern tokens:', tokens);
                console.log('Pattern color:', patternColor);
                
                // For attention heatmap, we can highlight the rows/columns corresponding to these tokens
                // This is a simplified approach - in a full implementation, you might want to highlight specific cells
            }
        } else {
        }
    }
    
    function highlightAttentionPattern(section, patternIndex) {
        if (!processData || !processData.tokens) return;
        
        // Extract tokens from the section text
        const sectionText = section.textContent;
        const tokens = [];
        
        // Find tokens mentioned in the section (look for quoted tokens)
        const tokenMatches = sectionText.match(/'([^']+)'/g);
        if (tokenMatches) {
            tokenMatches.forEach(match => {
                const token = match.replace(/'/g, '');
                if (processData.tokens.includes(token)) {
                    tokens.push(token);
                }
            });
        }
        
        // Highlight tokens in the attention heatmap
        if (tokens.length > 0 && window.Plotly) {
            const heatmapElement = document.getElementById('attention-heatmap');
            if (heatmapElement && heatmapElement.data) {
                // Create highlight effect by updating colors
                const update = {
                    'colorscale': [[0, '#f8f9fa'], [0.5, '#3498db'], [1, '#e74c3c']],
                    'zmin': 0,
                    'zmax': 1
                };
                
                Plotly.restyle('attention-heatmap', update);
            }
        }
    }
    
    function clearAttentionHighlight() {
        if (window.Plotly) {
            const heatmapElement = document.getElementById('attention-heatmap');
            if (heatmapElement && heatmapElement.data) {
                // Restore original colorscale
                const update = {
                    'colorscale': 'Viridis'
                };
                
                Plotly.restyle('attention-heatmap', update);
            }
        }
    }

    function showSampleGeneratorModal() {
        // Create a modal for sample generation form
        const modal = document.createElement('div');
        modal.className = 'sample-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>🎲 Generate Sample Text</h3>
                    <button class="close-modal">&times;</button>
                </div>
                <div class="modal-body">
                    <form id="sample-generator-form">
                        <div class="form-group">
                            <label for="sample-topic">Topic:</label>
                            <input type="text" id="sample-topic" placeholder="e.g., Artificial Intelligence, Cooking, Space Exploration" value="Artificial Intelligence">
                        </div>
                        <div class="form-group">
                            <label for="sample-complexity">Complexity:</label>
                            <select id="sample-complexity">
                                <option value="simple">Simple</option>
                                <option value="medium" selected>Medium</option>
                                <option value="complex">Complex</option>
                            </select>
                        </div>
                        <div class="form-actions">
                            <button type="button" class="cancel-btn" onclick="this.closest('.sample-modal').remove()">Cancel</button>
                            <button type="submit" class="generate-btn">Generate</button>
                        </div>
                    </form>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Add event listeners
        modal.querySelector('.close-modal').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        modal.querySelector('#sample-generator-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const topic = document.getElementById('sample-topic').value.trim();
            const complexity = document.getElementById('sample-complexity').value;
            
            if (!topic) {
                alert('Please enter a topic');
                return;
            }
            
            generateAndUseSample(topic, complexity, modal);
        });
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });
    }
    
    function generateAndUseSample(topic, complexity, modal) {
        const generateBtn = modal.querySelector('.generate-btn');
        const originalText = generateBtn.textContent;
        
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
        
        fetch('/generate-sample', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                topic: topic,
                complexity: complexity
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.sample_text) {
                // Insert the generated text into the input field
                document.getElementById('input-text').value = data.sample_text;
                // Close the modal
                document.body.removeChild(modal);
            } else {
                alert('Error generating sample: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error generating sample: ' + error.message);
        })
        .finally(() => {
            generateBtn.disabled = false;
            generateBtn.textContent = originalText;
        });
    }


    // Add CSS styles for AI features
    const geminiStyles = document.createElement('style');
    geminiStyles.textContent = `
        /* Input Section Styles */
        .input-container textarea {
            width: 100%;
            min-height: 100px;
            padding: 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.5;
            resize: vertical;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        
        .input-buttons {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .primary-btn, .secondary-btn {
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }
        
        .primary-btn {
            background-color: #28a745;
            color: white;
        }
        
        .primary-btn:hover {
            background-color: #218838;
        }
        
        .secondary-btn {
            background-color: #28a745;
            color: white;
        }
        
        .secondary-btn:hover {
            background-color: #218838;
        }
        
        /* Visualization Layout */
        .viz-row {
            display: block;
            margin-bottom: 20px;
        }
        
        .main-viz {
            width: 100%;
        }
        
        .plot-analysis-container {
            display: flex;
            gap: 20px;
            margin-top: 15px;
            width: 100%;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .plot-section {
            flex: 3;
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .plot-section #embedding-plot {
            width: 100%;
            height: 650px;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 0;
            overflow: hidden;
            box-sizing: border-box;
        }
        
        .plot-section #attention-heatmap {
            width: 100%;
            height: 650px;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 0;
            overflow: hidden;
            box-sizing: border-box;
        }
        
        .analysis-section {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            border-left: 4px solid #28a745;
            flex: 1;
            height: 650px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .analysis-content.scrollable {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding-right: 10px;
            max-height: 450px;
            min-height: 0;
        }
        
        .analysis-content .analysis-text {
            background-color: white;
            padding: 0;
            border-radius: 0;
            border: none;
            line-height: 1.6;
            margin-bottom: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .analysis-content .analysis-text h1,
        .analysis-content .analysis-text h2,
        .analysis-content .analysis-text h3,
        .analysis-content .analysis-text h4 {
            color: #2c3e50;
            margin-top: 20px;
            margin-bottom: 12px;
            font-weight: 700;
            font-size: 1.1em;
            border-bottom: none;
            padding-bottom: 0;
        }
        
        .analysis-content .analysis-text h1:first-child,
        .analysis-content .analysis-text h2:first-child,
        .analysis-content .analysis-text h3:first-child,
        .analysis-content .analysis-text h4:first-child {
            margin-top: 0;
        }
        
        .analysis-content .analysis-text p {
            margin-bottom: 12px;
            color: #34495e;
            font-size: 14px;
        }
        
        .analysis-content .analysis-text ul,
        .analysis-content .analysis-text ol {
            margin-left: 0;
            margin-bottom: 12px;
            padding-left: 20px;
        }
        
        .analysis-content .analysis-text li {
            margin-bottom: 6px;
            color: #34495e;
            font-size: 14px;
        }
        
        .analysis-content .analysis-text strong {
            font-weight: 600;
            color: #2c3e50;
        }
        
        .analysis-content .analysis-text code {
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            color: #e74c3c;
            font-size: 13px;
        }
        
        /* Specific styling to match the image exactly */
        .analysis-content .pattern-section {
            margin-bottom: 20px;
            padding: 12px;
            background-color: white;
            border-left: 4px solid #3498db;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .analysis-content .pattern-section:hover {
            background-color: #f8f9fa;
            border-left-color: #2980b9;
            transform: translateX(2px);
        }
        
        .analysis-content .pattern-section h3 {
            color: #2c3e50;
            margin: 0 0 8px 0;
            font-size: 1em;
            font-weight: 700;
            border-bottom: none;
            padding-bottom: 0;
        }
        
        .analysis-content .pattern-section .pattern-label {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 3px;
            font-size: 12px;
            display: block;
        }
        
        .analysis-content .pattern-section .pattern-content {
            color: #34495e;
            font-size: 12px;
            line-height: 1.4;
            margin-bottom: 6px;
            display: block;
        }
        
        .analysis-content .pattern-section .pattern-tokens {
            background-color: #ecf0f1;
            padding: 6px 10px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 11px;
            color: #2c3e50;
            margin-bottom: 6px;
            display: block;
        }
        
        /* Different colors for different clusters */
        .analysis-content .pattern-section:nth-child(1) { border-left-color: #e74c3c; }
        .analysis-content .pattern-section:nth-child(2) { border-left-color: #f39c12; }
        .analysis-content .pattern-section:nth-child(3) { border-left-color: #27ae60; }
        .analysis-content .pattern-section:nth-child(4) { border-left-color: #3498db; }
        .analysis-content .pattern-section:nth-child(5) { border-left-color: #9b59b6; }
        .analysis-content .pattern-section:nth-child(6) { border-left-color: #e67e22; }
        
        /* Cluster header styling */
        .cluster-header {
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 2px solid currentColor;
        }
        
        /* Clickable cluster styling */
        .cluster-clickable {
            cursor: pointer;
            transition: all 0.3s ease;
            border-radius: 4px;
            padding: 8px;
            margin: -8px;
        }
        
        .cluster-clickable:hover {
            background-color: #f8f9fa;
            transform: translateX(2px);
        }
        
        .pattern-clickable {
            cursor: pointer;
            transition: all 0.3s ease;
            border-radius: 4px;
            padding: 8px;
            margin: -8px;
        }
        
        .pattern-clickable:hover {
            background-color: #f8f9fa;
            transform: translateX(2px);
        }
        
        .highlight-hint {
            font-size: 11px;
            color: #7f8c8d;
            text-align: center;
            margin-top: 8px;
            padding: 4px;
            background-color: #ecf0f1;
            border-radius: 3px;
            font-style: italic;
        }
        
        /* Highlight button styling removed - using clickable clusters instead */
        
        
        /* Clear highlight button styling removed - no longer needed */
        
        /* Fullscreen functionality styling */
        .plot-header {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .fullscreen-btn {
            background: #6c757d;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s ease;
        }
        
        .fullscreen-btn:hover {
            background: #5a6268;
        }
        
        .fullscreen-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .fullscreen-content {
            width: 95%;
            height: 95%;
            background: white;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
        }
        
        .fullscreen-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            background: #f8f9fa;
            border-radius: 8px 8px 0 0;
        }
        
        .fullscreen-header h3 {
            margin: 0;
            color: #2c3e50;
        }
        
        .close-fullscreen-btn {
            background: #dc3545;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        
        .close-fullscreen-btn:hover {
            background: #c82333;
        }
        
        .fullscreen-plot-container {
            flex: 1;
            padding: 20px;
            overflow: hidden;
        }
        
        .analysis-content.scrollable::-webkit-scrollbar {
            width: 8px;
        }
        
        .analysis-content.scrollable::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        .analysis-content.scrollable::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        
        .analysis-content.scrollable::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        
        /* Ensure scrollbar is always visible when content overflows */
        .analysis-content.scrollable {
            scrollbar-width: thin;
            scrollbar-color: #888 #f1f1f1;
        }
        
        .analysis-panel h3 {
            color: #28a745;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .analysis-btn {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            margin-bottom: 15px;
            transition: background-color 0.2s ease;
            flex: 1;
            min-width: 0;
        }
        
        .analysis-btn:hover {
            background-color: #218838;
        }
        
        .analysis-btn:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        
        /* Model Selection */
        .model-selection {
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }
        
        .model-selection label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 500;
            color: #495057;
        }
        
        .model-selection input[type="checkbox"] {
            transform: scale(1.2);
        }
        
        /* Gemini Analysis Section */
        .gemini-analysis-section {
            margin: 20px 0;
        }
        
        .gemini-content {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #007bff;
        }
        
        .analysis-content, .gemini-insights {
            margin-bottom: 20px;
        }
        
        .analysis-content h4, .gemini-insights h4 {
            color: #007bff;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .analysis-text, .insights-text {
            background-color: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
            line-height: 1.6;
        }
        
        /* Markdown Content Styling */
        .markdown-content h1, .markdown-content h2, .markdown-content h3, .markdown-content h4 {
            color: #333;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        .markdown-content h1 { font-size: 1.5em; }
        .markdown-content h2 { font-size: 1.3em; }
        .markdown-content h3 { font-size: 1.2em; }
        .markdown-content h4 { font-size: 1.1em; }
        
        .markdown-content p {
            margin-bottom: 10px;
        }
        
        .markdown-content ul, .markdown-content ol {
            margin-left: 20px;
            margin-bottom: 10px;
        }
        
        .markdown-content li {
            margin-bottom: 5px;
        }
        
        .markdown-content strong {
            font-weight: bold;
            color: #007bff;
        }
        
        .markdown-content code {
            background-color: #f1f3f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        /* Sample Text Modal */
        .sample-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .modal-content {
            background-color: white;
            border-radius: 8px;
            max-width: 500px;
            max-height: 80vh;
            overflow-y: auto;
            margin: 20px;
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }
        
        .modal-header h3 {
            margin: 0;
            color: #333;
        }
        
        .close-modal {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #666;
        }
        
        .modal-body {
            padding: 20px;
        }
        
        /* Form Styles */
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #333;
        }
        
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            box-sizing: border-box;
        }
        
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }
        
        .form-actions {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
            margin-top: 20px;
        }
        
        .cancel-btn, .generate-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            transition: background-color 0.2s ease;
        }
        
        .cancel-btn {
            background-color: #6c757d;
            color: white;
        }
        
        .cancel-btn:hover {
            background-color: #545b62;
        }
        
        .generate-btn {
            background-color: #28a745;
            color: white;
        }
        
        .generate-btn:hover {
            background-color: #218838;
        }
        
        .generate-btn:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        
        /* Embedding Comparison */
        .embedding-comparison {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }
        
        .embedding-comparison h3 {
            color: #28a745;
            margin-bottom: 15px;
        }
        
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .metric-item {
            background-color: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .metric-label {
            font-weight: 500;
            color: #495057;
        }
        
        .metric-value {
            font-weight: bold;
            color: #007bff;
            font-size: 1.1em;
        }
        
        .warning-message {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 20px;
            color: #856404;
        }
        
        .warning-message h4 {
            color: #856404;
            margin-bottom: 10px;
        }
        
        .warning-message a {
            color: #007bff;
            text-decoration: none;
        }
        
        .warning-message a:hover {
            text-decoration: underline;
        }
        
        /* Responsive Design */
        @media (max-width: 1024px) {
            .analysis-panel {
                margin-top: 20px;
                max-height: 350px;
            }
            
            .analysis-content.scrollable {
                max-height: 250px;
            }
        }
        
        @media (max-width: 768px) {
            .input-buttons {
                flex-direction: column;
                gap: 10px;
            }
            
            .plot-analysis-container {
                flex-direction: column;
            }
            
            .analysis-section {
                height: 500px;
                width: 100%;
            }
            
            .plot-section #embedding-plot {
                height: 500px;
                padding: 0;
                overflow: hidden;
            box-sizing: border-box;
        }
        
        .plot-section #attention-heatmap {
            height: 500px;
            padding: 0;
            overflow: hidden;
            box-sizing: border-box;
        }
            
            .analysis-content.scrollable {
                max-height: 250px;
            }
            
            .comparison-grid {
                grid-template-columns: 1fr;
            }
            
            .metric-item {
                flex-direction: column;
                align-items: flex-start;
                gap: 5px;
            }
            
            .modal-content {
                margin: 10px;
                max-height: 90vh;
            }
        }
    `;
    document.head.appendChild(geminiStyles);
    
});

// Update Plotly CDN warning
window.addEventListener('load', function() {
    console.log('Note: Using Plotly version 1.58.5. This is expected and won\'t affect functionality.');
});
