import gradio as gr
import tensorflow as tf
import keras
from sentinet_modules import SentenceEmbedding,custom_prep

MODEL_PATH = '../saved_models/sentiment_model.keras'

model = keras.models.load_model(MODEL_PATH, custom_objects={"SentenceEmbedding": SentenceEmbedding, "custom_prep": custom_prep})

def sentimdb_app(input_text):
    SAMPLE = tf.constant([input_text])
    prob = model.predict([SAMPLE])
    prob = round(prob[0][0],ndigits=4)
    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = (prob * 100) if label=="Positive" else (1-prob)*100
    return label , confidence

senti_demo = gr.Interface(fn=sentimdb_app, inputs=[gr.components.Textbox(label="Review Text")],
                          outputs=[gr.components.Textbox(label="Sentiment"), gr.components.Textbox(label="Confidence")],
                        title="SentiNet", theme="default", show_api=False, 
                         description= "SentiNet is a sentiment analysis system powered by deep learning, built on the Large Movie Review Dataset (https://ai.stanford.edu/~amaas/data/sentiment/). This project leverages TensorFlow's data API for efficient data loading, implements custom sentence embeddings trained from scratch, and uses an MLP for classification. The model is developed with Keras, incorporating advanced training techniques like Nesterov optimization, kernel weight decay, dropout, early stopping, and exponential learning rate scheduling. The final model is ready for deployment with a Gradio web UI and Docker support for seamless installation. Projct Link: github.com/hoom4n/sentinet")
senti_demo.launch()