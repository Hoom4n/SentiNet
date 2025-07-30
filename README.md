# SentiNet
  
SentiNet is a sentiment analysis system powered by deep learning, built on the Large Movie Review Dataset (https://ai.stanford.edu/~amaas/data/sentiment/). This project leverages TensorFlow's data API for efficient data loading, implements custom sentence embeddings trained from scratch, and uses an MLP for classification. The model is developed with Keras, incorporating advanced training techniques like Nesterov optimization, kernel weight decay, dropout, early stopping, and exponential learning rate scheduling. The final model is ready for deployment with a Gradio web UI and Docker support for seamless installation.
This project builds a sentiment analysis system using two core components: a custom TensorFlow Data API loader to process text reviews into a tf.data.Dataset, and a Keras-based pipeline for preprocessing, encoding, embedding, and classifying text reviews.


## ✅ End-to-End Sentiment Analysis Pipeline
Comprehensive workflow covering everything from efficient streaming data ingestion and preprocessing to text encoding, model training, classification, and web-based deployment.

- **Optimized TensorFlow Data Pipeline**  
  Built using TensorFlow's tf.data API, the pipeline supports automatic label inference from directory structure and improves training efficiency through prefetching, shuffling, and dynamic batching.
- **Custom Sentence Embedding Layer with Pretraining**  
  Developed a custom Keras layer to compute sentence embeddings by averaging token embeddings, normalized by sentence length—enabling more stable and interpretable feature representations. Includes optional pretraining for enhanced embedding quality.
- **Advanced Training Techniques**  
  Accelerated model convergence with Nesterov-accelerated Adam (Nadam), regularized using weight decay, dropout layers, and early stopping to prevent overfitting and enhance generalization performance.
- **Flexible Deployment Options** 
  Deployed across multiple platforms including a Hugging Face Spaces demo, an interactive Gradio web app, and a Dockerized setup for seamless local or cloud deployment.


## 🛠️ Installation 

### Option 1: Online Demo

You can try the online demo on HuggingFace Spaces: <a href="https://huggingface.co/spaces/hoom4n/SentiMDBd">https://huggingface.co/spaces/hoom4n/SentiMDB</a>

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
