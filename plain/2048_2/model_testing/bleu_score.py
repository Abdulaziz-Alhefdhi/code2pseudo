from keras.models import load_model
from utils import retrieve_texts, DataObject, data_shapes, shape_info, token_integer_mapping, decode_sequence
import numpy as np
from tqdm import tqdm
import pickle
from nltk.translate.bleu_score import corpus_bleu


latent_dim = 1024     # Latent dimensionality of the encoding space.
num_samples = 20000  # Number of samples to train on.
max_input_length = 1000000  # Number of largest acceptable input length
max_target_length = 1800  # Number of largest acceptable target length

path_to_train_data = '../dataset/'
path_to_test_data = 'dataset/'

"""
# Get train data
input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens = \
    retrieve_texts(path_to_train_data, num_samples, max_input_length, max_target_length)
train_do = DataObject(input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens)
num_encoder_tokens_train, num_decoder_tokens_train, max_encoder_seq_length_train, \
max_decoder_seq_length_train, n_input_samples_train = data_shapes(train_do)
"""

# Get test data
input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens = \
    retrieve_texts(path_to_test_data, num_samples, max_input_length, max_target_length)
    ###retrieve_texts(path_to_test_data, num_samples, max_input_length, max_target_length)
test_do = DataObject(input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens)
num_encoder_tokens_test, num_decoder_tokens_test, max_encoder_seq_length_test, \
max_decoder_seq_length_test, n_input_samples_test = data_shapes(test_do)

# Print shape info
"""
print("================\nTraining data info:-")
shape_info(n_input_samples_train, num_encoder_tokens_train, num_decoder_tokens_train, max_encoder_seq_length_train, max_decoder_seq_length_train)
print("================\nTesting data info:-")
shape_info(n_input_samples_test, num_encoder_tokens_test, num_decoder_tokens_test, max_encoder_seq_length_test, max_decoder_seq_length_test)
print("================")
"""


references = []
for a_list in test_do.target_lists:
    references.append([a_list[1:-1]])

with open('predicted_lists.pkl', 'rb') as f:
    predicted_lists = pickle.load(f)

"""
for i in range(1000):
    print(predicted_lists[i])
    print(references[i][0])
import sys
sys.exit()
"""

score = corpus_bleu(references, predicted_lists)
print("Bleu score:", score)
