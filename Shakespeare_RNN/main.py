
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() # for better performance?

characters = sorted(set(text)) # filters unique characters

char_to_index = {c: i for i, c in enumerate(characters)}  # maps characters to indices
index_to_char = {i: c for i, c in enumerate(characters)}  # maps indices to characters

SEQUENCE_LENGTH = 40  # Length of each input sequence
STEP_SIZE = 3  # Step size for moving the window

sentences = []
next_chars = []

for i in range(0, len(text) - SEQUENCE_LENGTH, STEP_SIZE): # this lists sample texts and their next characters
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])

x = np.zeros((len(sentences), SEQUENCE_LENGTH, len(characters)), dtype=np.bool) 
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)


for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1    
    y[i, char_to_index[next_chars[i]]] = 1


# Building RNN model
model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

model.fit(x, y, batch_size=256, epochs=4)

# Save the model
model.save('shakespeare_rnn.keras')


model = tf.keras.models.load_model('shakespeare_rnn.keras')

# copy pasted code for sampling
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(length, temperature):
    start_index = np.random.randint(0, len(text) - SEQUENCE_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQUENCE_LENGTH]
    generated += sentence

    for i in range(length):
        x = np.zeros((1, SEQUENCE_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x[0, t, char_to_index[char]] = 1

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

print("--------0.1---------")
print(generate_text(400, 0.1))

print("--------1---------")
print(generate_text(400, 1))

print("--------1.5---------")
print(generate_text(400, 1.5))