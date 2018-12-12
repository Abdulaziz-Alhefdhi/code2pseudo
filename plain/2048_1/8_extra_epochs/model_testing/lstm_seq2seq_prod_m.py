from keras.models import load_model
from utils import retrieve_texts, DataObject, data_shapes, shape_info, token_integer_mapping, decode_sequence
import numpy as np
from tqdm import tqdm
import pickle


###latent_dim = 2048     # Latent dimensionality of the encoding space.
num_samples = 20000  # Number of samples to train on.
max_input_length = 1000000  # Number of largest acceptable input length
max_target_length = 1800  # Number of largest acceptable target length

path_to_train_data = '../dataset/'
path_to_test_data = 'dataset/'

# Get train data
input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens = \
    retrieve_texts(path_to_train_data, num_samples, max_input_length, max_target_length)
train_do = DataObject(input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens)
num_encoder_tokens_train, num_decoder_tokens_train, max_encoder_seq_length_train, \
max_decoder_seq_length_train, n_input_samples_train = data_shapes(train_do)

"""
for i in range(len(train_do.input_tokens)):
    if train_do.input_tokens[i] == "import":
        print(train_do.input_tokens[i-3])
        print(train_do.input_tokens[i-2])
        print(train_do.input_tokens[i-1])
        print(train_do.input_tokens[i])
        print(train_do.input_tokens[i+1])
import sys
sys.exit()
"""

# Get test data
input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens = \
    retrieve_texts(path_to_test_data, num_samples, max_input_length, max_target_length)
    ###retrieve_texts(path_to_test_data, num_samples, max_input_length, max_target_length)
test_do = DataObject(input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens)
num_encoder_tokens_test, num_decoder_tokens_test, max_encoder_seq_length_test, \
max_decoder_seq_length_test, n_input_samples_test = data_shapes(test_do)

# Print shape info
print("================\nTraining data info:-")
shape_info(n_input_samples_train, num_encoder_tokens_train, num_decoder_tokens_train, max_encoder_seq_length_train, max_decoder_seq_length_train)
print("================\nTesting data info:-")
shape_info(n_input_samples_test, num_encoder_tokens_test, num_decoder_tokens_test, max_encoder_seq_length_test, max_decoder_seq_length_test)
print("================")

# Replace unseen-before tokens with: <unknown>
# Find
###i, j = 0, 0
unseen_tokens = []
for token in test_do.input_tokens:
    ###i += 1
    if token not in train_do.input_tokens:
        ###j += 1
        unseen_tokens.append(token)
###print("================")
###print(i, j, str(j/i*100)+"%")
# Replace
for i in range(len(test_do.input_lists)):
    for j in range(len(test_do.input_lists[i])):
        if test_do.input_lists[i][j] in unseen_tokens:
            test_do.input_lists[i][j] = "<unknown>"

# Print list if it has <unknown> in it
###for item in test_do.input_lists:
   ### if "<unknown>" in item:
      ###  print(item)

# Converting tokens to integers (Neural Networks accept only integers as inputs), and
# reverse-lookup token index to decode sequences back to something readable.
###input_token_index, _, reverse_input_token_index, _ = token_integer_mapping(test_do)
input_token_index, target_token_index, reverse_input_token_index, reverse_target_token_index = \
    token_integer_mapping(train_do.input_tokens, train_do.target_tokens)
    ###token_integer_mapping(test_do.input_tokens, test_do.target_tokens)


# Load the trained model
model = load_model('../checkpoints/c2p_att_plain_scaled_extra_30plus-06.hdf5')
print("================")

###sample_limit = 10
sample_limit = n_input_samples_test
# Prepare input samples to be decoded
encoder_input_data = np.zeros((sample_limit, max_encoder_seq_length_test), dtype='int32')
# Fill up encoder_input_data with testing data
j = 0
for i, input_list in enumerate(test_do.input_lists):
    j += 1
    if j > sample_limit:
        break
    for t, token in enumerate(input_list):
        encoder_input_data[i, t] = input_token_index[token]

###for i in range(5):
   ### print(test_do.input_lists[i])
###import sys
###sys.exit()

"""
j = 0
for i, input_list in enumerate(train_do.input_lists):
    j += 1
    if j > sample_limit:
        break
    # Loop input sequences
    for t, token in enumerate(input_list):
        ###encoder_input_data[i, t] = input_token_index[token]
        encoder_input_data[i, t, 0] = 0
        encoder_input_data[i, t, input_token_index[token]] = 1
"""

# Test samples from the beginning of the testing dataset
###for seq_index in range(n_input_samples_train):
c = 1
predicted_lists = []
for seq_index in tqdm(range(sample_limit)):
    # Take one sequence (part of the training set) for trying out decoding.
    input_seq = encoder_input_data[seq_index:seq_index+1]
    ###print('Input sentence: ' + test_do.input_texts[seq_index])
    input_seq2 = encoder_input_data[seq_index]
    input_list_tbp = []
    for i in input_seq2:
        if i == 0:
            break
        input_token = reverse_input_token_index[i]
        input_list_tbp.append(input_token)
    to_print_out = ''
    for token in input_list_tbp:
        to_print_out += token + ' '
    ###print('Encoded sentence:\n' + to_print_out + '\n')

    ###print('Target sentence: ' + test_do.target_texts[seq_index])
    decoded_sentence = decode_sequence(input_seq, model, max_decoder_seq_length_train, target_token_index, reverse_target_token_index)
    predicted_lists.append(decoded_sentence)
    to_print_out2 = ''
    for token in decoded_sentence:
        to_print_out2 += token + ' '
    ###print('Decoded sentence: ' + to_print_out)
    ###print('-')

    # Write output to file
    with open("testing_output", "a") as f:
        f.write(str(c) + '.a) Input sentence:   ' + test_do.input_texts[seq_index] + "\n")
        f.write(str(c) + '.b) Encoded sentence: ' + to_print_out + "\n")
        f.write(str(c) + '.c) Target sentence:  ' + test_do.target_texts[seq_index] + "\n")
        f.write(str(c) + '.d) Decoded sentence: ' + to_print_out2 + "\n-\n")
    c += 1

# Save predicted lists to disk for result evaluation
with open('predicted_lists.pkl', 'wb') as f:  
    pickle.dump(predicted_lists, f)

