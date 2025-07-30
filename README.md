# SentiNet
  
SentiNet is a deep learning–based sentiment analysis system built on the Large Movie Review Dataset (https://ai.stanford.edu/~amaas/data/sentiment/). The project features a custom TensorFlow tf.data pipeline for efficient streaming of text data, along with a custom sentence embedding layer trained from scratch. Text reviews are processed and classified using a Keras-based MLP architecture, enhanced with advanced training techniques including Nesterov-accelerated optimization, kernel weight decay, dropout regularization, early stopping, and exponential learning rate scheduling.

The pipeline is modular and fully end-to-end—from data loading and preprocessing to encoding, embedding, and classification. The final model is deployable via a Gradio web UI and includes Docker support for seamless, reproducible installation. Repo features a comperhensive <a href="https://nbviewer.org/github/hoom4n/SentiNet/blob/main/SentiNet.ipynb">Jupyter Notebook</a>

## ✅ End-to-End Sentiment Analysis Pipeline
Comprehensive workflow covering everything from efficient streaming data ingestion and preprocessing to text encoding, model training, classification, and web-based deployment.

- **High-Performance TensorFlow Data Pipeline**  
  Built with a custom tf.data pipeline designed for large-scale text classification tasks. Automatically infers labels from directory structure, with dynamic support for training/validation splits. Optimized using parallel file reading, caching, shuffling, prefetching, and batching—resulting in a highly efficient and GPU-ready input pipeline. Key features include:
  - *Label Inference via Directory Structure:* Automatically detects class labels from folder names without manual labeling.
  - *Dynamic Train/Validation Split:* Enables easy experimentation with customizable splits using simple flags.
  - *Memory-Efficient Streaming:* Utilizes AUTOTUNE, caching, and prefetching to minimize latency and maximize throughput.
  - *Customizable Pipeline Parameters:* Offers full control over batch size, shuffle buffer, prefetch depth, and seed for reproducibility.
- **Custom Sentence Embedding Layer with Pretraining**  
  Developed a custom Keras layer to compute sentence embeddings by averaging token embeddings, normalized by sentence length—enabling more stable and interpretable feature representations. Includes optional pretraining for enhanced embedding quality.
- **Advanced Training Techniques**  
  Accelerated model convergence with Nesterov-accelerated Adam (Nadam), regularized using weight decay, dropout layers, and early stopping to prevent overfitting and enhance generalization performance.
- **Flexible Deployment Options** 
  Deployed across multiple platforms including a Hugging Face Spaces demo, an interactive Gradio web app, and a Dockerized setup for seamless local or cloud deployment.


## 🛠️ Installation 

### Option 1: Online Demo

You can try the online demo on HuggingFace Spaces: <a href="https://huggingface.co/spaces/hoom4n/SentiNet">https://huggingface.co/spaces/hoom4n/SentiNet</a>

### Option 2: Local Setup

```bash
# clone project repository
git clone https://github.com/hoom4n/SentiNet.git
cd SentiNet

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # On Windows

pip install -r requirements.txt

# Run Gradio application
python app/app.py
```

Access applicaion at: http://localhost:7861

### Option 3: Docker Deployment

```bash
# clone project repository
git clone https://github.com/hoom4n/SentiNet.git
cd SentiNet

# Build image and start container for the first time
docker compose up --build

# For subsequent runs
docker compose up
```

Access applicaion at: http://localhost:8080
