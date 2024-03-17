#####################################################
# TextTransformerClassifier_Vanilla.py
# 
# Program extended from https://keras.io/examples/nlp/text_classification_with_transformer/
#
# This program classifies sentence reviews using a Transformer trained from scratch.
# The data is loaded directly from the data files instead of the pre-built from Keras (as 
# originally implemented) to provide a better insight on how to do that, which can be
# useful for its application to other datasets. See link above for further details.
#
# Version: 1.1 -- simplified and more organised than an earlier version
# Date: 06 March 2024
# Contact: hcuayahuitl@lincoln.ac.uk
#####################################################

import os
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from official.nlp import optimization

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output, attn_scores = self.att(inputs, inputs, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def get_dictionary(file_path, vocab_size):
    print("EXTRACTING dictionary from "+str(file_path))
    # step 1: generate a dictionary of words with their frequencies
    word_freqs = {}
    file_counter = 0
    folders = os.listdir(file_path)
    for folder in folders:
        files = os.listdir(os.path.join(file_path, folder))
        for fileName in files:
            file_counter += 1
            with open(os.path.join(file_path, folder, fileName), 'rt') as f:
                for line in f:
                    line = re.sub(r'[^\w\s]', '', line) # remove punctuation
                    for word in line.split():
                        if word in word_freqs:
                            counter = word_freqs[word]
                            word_freqs[word] = counter+1
                        else:
                            word_freqs[word] = 1

    # step 2: generate a set of word-index (and index-word) pairs -- the most popular words
    words = {}
    indexes = {}
    num_words = 1
    sorted_word_freqs = sorted(word_freqs.items(), key=lambda x:x[1], reverse=True)
    for word_count in sorted_word_freqs:
        word = word_count[0]
        count = word_count[1]
        words[word] = num_words
        indexes[num_words] = word
        num_words += 1
        if num_words > vocab_size:
            break 
    print("|words|=%s |indexes|=%s" % (len(words), len(indexes)))
    return words, indexes

def get_dataset(file_path, words, indexes, maxlen):
    print("READING folder="+str(file_path))
    x_train = []
    y_train = []
    folders = os.listdir(file_path)
    for folder in folders:
        label = [1, 0] if folder == 'pos' else [0, 1]
        files = os.listdir(file_path+"/"+folder)
        for fileName in files:
            sentence_vector = []
            with open(os.path.join(file_path, folder, fileName), 'rt') as f:
                for line in f:
                    line = re.sub(r'[^\w\s]', '', line) # remove punctuation
                    num_words = 0
                    for word in line.split():
                        if num_words >= maxlen: break
                        index = words[word] if word in words else 0
                        sentence_vector.append(int(index))
                        num_words += 1
                    for i in range(len(sentence_vector), maxlen):
                        sentence_vector.append(0) # zero-padding
            x_train.append(sentence_vector)
            y_train.append(label)

    x_train = tf.stack(x_train) # sentences in numeric format
    y_train = tf.stack(y_train) # labels
    return x_train, y_train

def print_example_sentences(x_train, x_test):
    print("EXAMPLE training/test sentences...")
    for i in range(0, 3):
        print("TRAIN: %s -> %s" % (x_train[i], len(x_train[i])))
    for i in range(0, 3):
        print("TEST: %s -> %s" % (x_test[i], len(x_test[i])))
		
def create_classifier_model(vocab_size, maxlen):
    embed_dim = 64 # Embedding size for each token
    num_heads = 2 # Number of attention heads
    ff_dim = 32 # Hidden layer size in feed forward network inside transformer

    print("CREATING classifier model...")
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def run_experiment(vocab_size, maxlen, datapath_train, datapath_test):
    words, indexes = get_dictionary(datapath_train, vocab_size)
    x_train, y_train = get_dataset(datapath_train, words, indexes, maxlen)
    x_test, y_test = get_dataset(datapath_test, words, indexes, maxlen)
    print("x_train shape: %s y_train shape: %s" %(x_train.shape, y_train.shape))
    print("x_test shape: %s y_test shape: %s" % (x_test.shape, y_test.shape))
    print_example_sentences(x_train, x_test)

    epochs = 10
    batch_size = 32
    num_train_steps = (len(x_train) / batch_size) * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    optimizer = optimization.create_optimizer(init_lr=3e-5,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
    loss = tf.keras.losses.BinaryCrossentropy()#from_logits=True)
    metrics = tf.keras.metrics.BinaryAccuracy()
    
    model = create_classifier_model(vocab_size, maxlen)

    print("TRAINING & EVALUATING classifier model...")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    _, accuracy = model.evaluate(x_test, y_test)
    print("Test Accuracy: %s" % (accuracy))


datapath_train = "datasets4transformers/aclImdb/train"
datapath_test = "datasets4transformers/aclImdb/test"
vocab_size = 20000 # Only consider the most frequent 20k words
maxlen = 50 # Only consider the first 200 words of each movie review
run_experiment(vocab_size, maxlen, datapath_train, datapath_test)
