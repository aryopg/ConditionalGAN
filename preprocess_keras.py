import os
import pickle
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.callbacks import EarlyStopping

os.environ['KERAS_BACKEND']='tensorflow'

processed_data_dir = 'data_keras'
train_data_dir = os.path.join(processed_data_dir, 'train')
test_data_dir = os.path.join(processed_data_dir, 'test')
if not os.path.exists(processed_data_dir):
    os.mkdir(processed_data_dir)
if not os.path.exists(train_data_dir):
    os.mkdir(train_data_dir)
if not os.path.exists(test_data_dir):
    os.mkdir(test_data_dir)

motions = []
claims = []

X = []
Y = []
Z = []
with open('dataset/claims.txt','r') as k:
	for idx,row in enumerate(k.readlines()):
		if idx > 0:
			var = row.split('\t')
			motions.append(var[0].lower())
			claims.append(var[2].lower())

with open('dataset/evidence.txt','r') as k:
	for idx,row in enumerate(k.readlines()):
		if idx > 0:
			var = row.split('\t')
			motions.append(var[0].lower())
			claims.append(var[2].lower())

print('Found %s motions.' % len(motions))
print('Found %s claims.' % len(claims))

if not os.path.isfile(os.path.join(processed_data_dir, 'lexicon-dict.pkl')):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(motions+claims)
    word_index = tokenizer.word_index
    inverted_word_dict = dict([[v,k] for k,v in word_index.items()])
    print('Total %s unique tokens.' % len(word_index))
    pickle.dump(word_index, open(os.path.join(processed_data_dir, 'lexicon-dict.pkl'), 'wb'))
    pickle.dump(inverted_word_dict, open(os.path.join(processed_data_dir, 'lexicon-dict-inverse.pkl'), 'wb'))
else:
    word_index = pickle.load(open(os.path.join(processed_data_dir, 'lexicon-dict.pkl'), 'rb'))
    inverted_word_dict = pickle.load(open(os.path.join(processed_data_dir, 'lexicon-dict-inverse.pkl'), 'rb'))
    lexicon_count = len(word_index)

for sentence in motions:
	# sentence_list = word_tokenize(sentence)
	sentence_list = text_to_word_sequence(sentence)
	data = []
	for word in sentence_list:
		data.append(word_index[word.lower()])
	X.append(data)

for sentence in claims:
	# sentence_list = word_tokenize(sentence)
	sentence_list = text_to_word_sequence(sentence)
	data = []
	for word in sentence_list:
		data.append(word_index[word.lower()])
	Y.append(data)

X = np.asarray(X)
Y = np.asarray(Y)

X = pad_sequences(X,maxlen=20, padding='post')
Y = pad_sequences(Y,maxlen=20, padding='post')

encoded_data = []
for i in range(len(X)):
    encoded_data.append({
        'input': {
            'encoded': X[i],
            'ori': motions[i]
        },
        'output': {
            'encoded': Y[i],
            'ori': claims[i]
        }
    })
if len(os.listdir(train_data_dir)) <= 0 and len(os.listdir(test_data_dir)) <= 0:
    seed = 7
    np.random.seed(seed)
    np.random.shuffle(encoded_data)
    # encoded_data = encoded_data[indices]
    nb_validation_samples = int(0.1 * len(encoded_data))
    data_train = encoded_data[:-nb_validation_samples]
    data_test = encoded_data[-nb_validation_samples:]
    for idx, data in enumerate(data_train):
        pickle.dump(data, open(os.path.join(train_data_dir, 'encoded_input_output' + '_' + str(idx+1) + '.pkl'), 'wb'))

    print("\rSaving Input-Output Training Data ... Done!")
    for idx, data in enumerate(data_test):
        pickle.dump(data, open(os.path.join(test_data_dir, 'encoded_input_output' + '_' + str(idx+1) + '.pkl'), 'wb'))

    print("\rSaving Input-Output Testing Data ... Done!")
