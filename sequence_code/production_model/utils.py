import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Activation, dot, concatenate, TimeDistributed


# Tokenize the data.
def tokenize_data(data_path, num_samples, max_input_length, max_target_length):
    # Variable declaration
    input_texts = []
    target_texts = []
    input_tokens = set()
    target_tokens = set()
    with open(data_path + 'code_seqs.txt', 'r', encoding='utf-8') as f:
        input_lines = f.read().split('\n')
    with open(data_path + 'javadocs.txt', 'r', encoding='utf-8') as f:
        target_lines = f.read().split('\n')
    input_lines = input_lines[: min(num_samples, len(input_lines) - 1)]
    target_lines = target_lines[: min(num_samples, len(target_lines) - 1)]
    for input_text, target_text in zip(input_lines, target_lines):
        # Throwing out long texts
        if len(input_text) > max_input_length or len(target_text) > max_target_length:
            continue
        # Replace problematic characters that're basically spaces!
        for char in ['\u2009', '\u202f', '\xa0']:
            target_text = target_text.replace(char, ' ')
        # Accept only letters and numbers for target texts
        ###for char in target_text:
        ### if not char.isdigit() and not char.isalpha():
        ###  target_text = target_text.replace(char, ' ')
        # We use "tab" as the "start sequence" character for the targets, and "\n" as "end sequence" character.
        ###target_texts.append('\t' + target_text + '\n')
        input_texts.append(input_text)
        target_texts.append(target_text)
        # Dealing with sequence-specific characters
        input_text = input_text.replace("(", " ( ")
        input_text = input_text.replace(")", " ) ")
        input_text = input_text.replace("_", " _ ")
        for token in input_text.split():
            input_tokens.add(token)
        for char in target_text:
            if (not char.isalpha()) and (char != " "):
                target_text = target_text.replace(char, " " + char + " ")
        for token in target_text.split():
            target_tokens.add(token.lower())
        # \t & \n are ignored in split(). Add them manually
        ###target_tokens.add('\t')
        ###target_tokens.add('\n')

    # Return desired data, with tokens being sorted and converted to lists
    return sorted(list(input_tokens)), sorted(list(target_tokens)), input_texts, target_texts


# To solve tokenization issues, convert sentences into lists of words
def list_texts(texts):
    lists = []
    for text in texts:
        a_list, word = [], ''
        for i, char in enumerate(text):
            if char.isalpha():
                word += char
                if (i == len(text)-1) or (not text[i+1].isalpha()):
                    a_list.append(word.lower())
            else:
                if char != ' ':
                    a_list.append(char)
                word = ''
        lists.append(a_list)
    return lists


# Get data
def retrieve_texts(data_path, num_samples, max_input_length, max_target_length):
    input_tokens, target_tokens, input_texts, target_texts = tokenize_data(data_path, num_samples, max_input_length, max_target_length)
    # Special treatment for input code sequences
    input_lists = []
    for txt in input_texts:
        txt = txt.replace("(", " ( ")
        txt = txt.replace(")", " ) ")
        txt = txt.replace("_", " _ ")
        input_lists.append(txt.split())
    target_lists = list_texts(target_texts)
    # Special considerations:
    # add "<unknown>" token for unknown words during testing, "<nothing>" token for padding during training, and "<eol>"
    # to indicate the end of input/target list
    input_tokens = input_tokens + ["<unknown>"]
    target_tokens = target_tokens + ["\n", "\t", "<unknown>"]
    for i in range(len(target_lists)):
        target_lists[i] = ["\t"] + target_lists[i] + ["\n"]

    return input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens
    ###return clipped_itxts, clipped_ttxts, clipped_ilists, clipped_tlists, input_tokens, target_tokens


class DataObject:
    def __init__(self, input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.input_lists = input_lists
        self.target_lists = target_lists
        self.input_tokens = input_tokens
        self.target_tokens = target_tokens
        # this will track the progress of the batches sequentially through the data set - once the data reaches the end of the data set it will reset back to zero
        self.current_idx = 0

    def generate(self, batch_size, max_encoder_seq_length, max_decoder_seq_length, num_decoder_tokens,
                 input_token_index, target_token_index):
        # Define input & output data and initialize them with zeros
        encoder_input_data = np.zeros((batch_size, max_encoder_seq_length), dtype='int32')
        decoder_input_data = np.zeros((batch_size, max_decoder_seq_length), dtype='int32')
        decoder_target_data = np.zeros((batch_size, max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        # Special initialization procedure for decoder_target_data
        for sample in decoder_target_data:
            for token in sample:
                token[0] = 1.
        while True:
            # fill input data & one-hot encode targets
            # Loop samples
            for i in range(batch_size):
                if (self.current_idx + i) >= len(self.input_lists):
                    print("\nA full iteration through the dataset has been completed. Last target sample # = " + str(
                        self.current_idx + i))
                    ###print("Last Target Sequence: " + str(self.target_lists[self.current_idx+i-1]))
                    self.current_idx = -i
                if (self.current_idx + i) % 2000 == 0:
                    if self.current_idx + i == 0:
                        print("\nBeginning of dataset..")
                    ###print("\nCurrent target sample # = " + str(self.current_idx + i))
                    ###print("Current Target Sequence: " + str(self.target_lists[self.current_idx + i]))
                # Loop input sequences
                for t, token in enumerate(self.input_lists[self.current_idx + i]):
                    encoder_input_data[i, t] = input_token_index[token]
                # Loop target sequences
                for t, token in enumerate(self.target_lists[self.current_idx + i]):
                    # decoder_target_data is ahead of decoder_input_data by one timestep
                    decoder_input_data[i, t] = target_token_index[token]
                    if t > 0:
                        # decoder_target_data will be ahead by one timestep and will not include the start character. Initial value altered.
                        decoder_target_data[i, t - 1, 0] = 0.
                        decoder_target_data[i, t - 1, target_token_index[token]] = 1.
            self.current_idx += batch_size
            yield ([encoder_input_data, decoder_input_data], decoder_target_data)


def data_shapes(do):
    num_encoder_tokens = len(do.input_tokens)
    num_decoder_tokens = len(do.target_tokens)
    max_encoder_seq_length = max([len(txt) for txt in do.input_lists])
    max_decoder_seq_length = max([len(txt) for txt in do.target_lists])
    n_input_samples = len(do.input_lists)

    return num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, n_input_samples

def shape_info(n_input_samples, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length):
    print('Number of samples:', n_input_samples)
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

def token_integer_mapping(input_tokens, target_tokens):
    input_token_index = dict([(token, i+1) for i, token in enumerate(input_tokens)])
    target_token_index = dict([(token, i+1) for i, token in enumerate(target_tokens)])
    reverse_input_token_index = dict((i, token) for token, i in input_token_index.items())
    reverse_target_token_index = dict((i, token) for token, i in target_token_index.items())
    return input_token_index, target_token_index, reverse_input_token_index, reverse_target_token_index


def build_model(latent_dim, num_encoder_tokens, num_decoder_tokens):
    encoder_inputs = Input(shape=(None,))
    encoder = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
    encoder = LSTM(latent_dim, return_sequences=True)(encoder)
    encoder_last = encoder[:, -1, :]

    decoder_inputs = Input(shape=(None,))
    decoder = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)(decoder_inputs)
    decoder = LSTM(latent_dim, return_sequences=True)(decoder, initial_state=[encoder_last, encoder_last])

    attention = dot([decoder, encoder], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)
    context = dot([attention, encoder], axes=[2, 1])
    decoder_combined_context = concatenate([context, decoder])
    output = TimeDistributed(Dense(latent_dim, activation="tanh"))(decoder_combined_context)
    decoder_outputs = TimeDistributed(Dense(num_decoder_tokens, activation="softmax"))(output)

    return encoder_inputs, decoder_inputs, decoder_outputs


    """
    # Set up the encoder. Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    en_x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
    _, state_h, state_c = LSTM(latent_dim, return_state=True)(en_x)
    # We discard 'encoder_outputs' and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder with attention, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None,))
    de_x = Embedding(num_decoder_tokens, latent_dim)
    final_de_x = de_x(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(final_de_x, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    return encoder_inputs, decoder_inputs, decoder_outputs, encoder_states, de_x, decoder_lstm, decoder_dense
    ###return encoder_inputs, decoder_outputs
    """


def inference_model(latent_dim, encoder_inputs, encoder_states, de_x, decoder_inputs, decoder_lstm, decoder_dense):
    # Encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # Decoder
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    final_dex2 = de_x(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

    return encoder_model, decoder_model


###def decode_sequence(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index,
   ###                 reverse_target_token_index):
def decode_sequence(input_seq, model, max_decoder_seq_length, target_token_index, reverse_target_token_index):
    """
    # Encode the input as state vectors.
    ###states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    ###target_seq = np.zeros((1, 1), dtype='int32')
    target_seq = np.zeros(shape=(len(input_seq), max_decoder_seq_length))
    # Populate the first character of target sequence with the start character.
    ###target_seq[0, 0] = target_token_index['\t']
    target_seq[:, 0] = target_token_index['\t']
    for i in range(1, max_decoder_seq_length):
        prediction = model.predict([input_seq, target_seq]).argmax(axis=2)
        if reverse_target_token_index[prediction[:, i][0]] == "\n":
            break
        target_seq[:, i] = prediction[:, i]
    ###print(target_seq[:, 1:])
    decoded_sentence = []
    for idx in target_seq[:, 1:][0]:
        if idx == 0:
            break
        decoded_sentence.append(reverse_target_token_index[idx])
    return decoded_sentence


    # Sampling loop for a batch of sequences (to simplify, here we assume a batch of size 1).
    
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        ###output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        output_tokens = model.predict([input_seq, target_seq])
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_token_index[sampled_token_index]
        decoded_sentence.append(sampled_token)
        print(decoded_sentence)
        print(len(decoded_sentence))
        # Exit condition: either hit max length or find stop character.
        if (sampled_token == '\n') or (len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1), dtype='int32')
        target_seq[0, 0] = target_token_index[sampled_token]
        # Update states
        ###states_value = [h, c]
    return decoded_sentence
    """
    target_seq = np.zeros(shape=(len(input_seq), max_decoder_seq_length))
    # Populate the first character of target sequence with the start character.
    target_seq[:, 0] = target_token_index['\t']
    for i in range(1, max_decoder_seq_length):
        prediction = model.predict([input_seq, target_seq]).argmax(axis=2)
        if reverse_target_token_index[prediction[:, i][0]] == "\n":
            break
        target_seq[:, i] = prediction[:, i]
    decoded_sentence = []
    for idx in target_seq[:, 1:][0]:
        if idx == 0:
            break
        decoded_sentence.append(reverse_target_token_index[idx])
    return decoded_sentence