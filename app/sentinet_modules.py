import tensorflow as tf
import keras

def custom_prep(text):
    """Preprocesses text by normalizing and cleaning it"""
    text = tf.strings.lower(text)  # lowercase
    text = tf.strings.regex_replace(text, pattern=r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
                                    , rewrite=" URL ") # remove urls
    text = tf.strings.regex_replace(text, pattern=r'(\#{1,6}\s*)|(\*{1,2})|(__)|(\~\~)|`{1,3}|!\[.*?\]\(.*?\)|\[.*?\]\(.*?\)|<.*?>', rewrite=" ") # remove markdowns
    text = tf.strings.regex_replace(text, pattern=r'\n', rewrite=" ")  # remove newlines
    text = tf.strings.regex_replace(text, pattern=r'[^\w\s]', rewrite=" ")  # remove punctuation
    text = tf.strings.regex_replace(text, pattern=r'\d+', rewrite=" NUMBER ")  # remove Numbers
    text = tf.strings.regex_replace(text, pattern="nbsp" , rewrite=" ")
    stopwords = ['the', 'a', 'an', 'of', 'on', 'in', 'at', 'for', 'to',
     'and', 'or', 'but', 'with', 'by', 'from', 'as', 'that',
     'this', 'these', 'those', 'which', 'who', 'whom', 'whose',
     'there', 'here', 'when', 'where', 'why', 'how']
    pattern = r'\b(' + '|'.join(stopwords) + r')\b'
    text = tf.strings.regex_replace(text,pattern=pattern , rewrite=" ") # remove stopwords
    text = tf.strings.regex_replace(text, pattern=r'\s+', rewrite=" ")  # Whitespace collapse
    return tf.strings.strip(text)


class SentenceEmbedding(keras.layers.Layer):
    """Custom layer for sentence embeddings using word embeddings and mean pooling.
    
    This layer converts tokenized text into dense sentence embeddings by:
    1. Applying an embedding layer to convert tokens into dense vectors.
    2. Computing the mean of word embeddings across the sequence.
    3. Scaling the mean by the square root of non-zero token counts to account for sequence length.
    
    Args:
        vocab_size: Integer, size of the vocabulary for the embedding layer.
        embedding_dim: Integer, dimensionality of the embedding vectors.
        **kwargs: Additional arguments passed to the parent Layer class.
    
    Raises:
        ValueError: If vocab_size or embedding_dim is not a positive integer.
    
    Returns:
        Tensor of shape (batch_size, embedding_dim) representing sentence embeddings.
    
    Example:
        ```python
        layer = SentenceEmbedding(vocab_size=10000, embedding_dim=128)
        inputs = tf.constant([[1, 2, 0], [3, 4, 5]])  # Tokenized input
        outputs = layer(inputs)  # Sentence embeddings
        ```
    """
    def __init__(self, vocab_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        if vocab_size <= 0 or embedding_dim <= 0:
            raise ValueError("vocab_size and embedding_dim must be positive integers")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    
    def call(self, X):
        embeddings = self.embedding(X)
        mean = keras.ops.mean(embeddings, axis=1)  
        reviews_word_count = keras.ops.count_nonzero(X, axis=1) 
        sqrt_root = keras.ops.sqrt(keras.ops.cast(reviews_word_count, "float32") + 1e-10) 
        return mean * keras.ops.expand_dims(sqrt_root, axis=-1)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "vocab_size": self.vocab_size, "embedding_dim": self.embedding_dim}