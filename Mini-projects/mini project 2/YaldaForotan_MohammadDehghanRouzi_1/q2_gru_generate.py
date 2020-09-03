from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, Embedding, GRU, SimpleRNN
import keras
import tensorflow as tf

''' Load dataset '''

path = 'shakespeare.txt'
txt = open(path).read()
len(txt)

characters = sorted(list(set(txt)))

import numpy as np

char2ind = dict((c, i) for i, c in enumerate(characters))
ind2char = dict((i, c) for i, c in enumerate(characters))
print(char2ind)
print(ind2char)

text_as_int = np.array([char2ind[c] for c in txt])
print(text_as_int.shape)

''' Load model '''
BATCH_SIZE = 1
embedding_dim = 256
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(65, embedding_dim, batch_input_shape=[BATCH_SIZE, None]),
    tf.keras.layers.GRU(1024, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(65)])
webs = 'my_model.h5'
# mine = 'checkpoint_1024_epoch_3.hdf5'
model.load_weights(webs)
model.build(tf.TensorShape([1, None]))
model.summary()

''' Test model '''
sample_txt = 'How are you?'  # I am good. What are you doing now?\n Actually I am a student at the university'
print(sample_txt)

sample_txt_list = np.zeros(len(sample_txt), dtype=int)
for i in range(len(sample_txt)):
    sample_txt_list[i] = char2ind[sample_txt[i]]


''' Web model '''
num_generate = 150
start_string = sample_txt

input_eval = [char2ind[s] for s in start_string]
input_eval = tf.expand_dims(input_eval, 0)

text_generated = []

for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)

    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(ind2char[predicted_id])
result = (start_string + ''.join(text_generated))
print(result)
