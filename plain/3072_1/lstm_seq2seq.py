from utils import retrieve_texts, data_shapes, shape_info, DataObject, token_integer_mapping, build_model
from keras.models import Model
###from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import datetime
from keras.models import load_model


batch_size = 32      # Batch size for training.
epochs = 1          # Number of epochs to train for.
latent_dim = 3072     # Latent dimensionality of the encoding space.
num_samples = 20000  # Number of samples to train on.
###num_samples = 2000  # Number of samples to train on.
data_path = 'dataset/'  # Path to the data txt files on disk.
###data_path = '/home/aa043/sea/data/code2pseudo/reduced_surface/train/'  # Path to the data txt files on disk.
# Maximum length for inputs and outputs (in terms of characters, not tokens)
max_input_length = 1000000 # Number of largest acceptibale input length
max_target_length = 1800 # Number of largest acceptibale target length
###max_input_length = 6000
###max_target_length = 1000000

# Get data
input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens = retrieve_texts(data_path, num_samples, max_input_length, max_target_length)
do = DataObject(input_texts, target_texts, input_lists, target_lists, input_tokens, target_tokens)
# Data shapes
num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, n_input_samples = data_shapes(do)
# Print shape info
print("================\nTraining data info:-")
shape_info(n_input_samples, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length)
print("Batch size:", batch_size)
print("Embedding dimensionality:", latent_dim)
print("================")

#for txt in do.input_texts:
 #   if "760" in txt:
  #      print(txt)

# Print longest input sample
###for i in range(len(do.input_lists)):
   ### if len(do.input_lists[i]) == max_encoder_seq_length:
      ###  print(do.input_texts[i])
        ###print(len(do.input_texts[i]))

"""
# Print percentage of removed data
i = 0
for input_text in do.input_texts:
	if len(input_text) > 6000:
		i += 1
print(i, len(do.input_texts), '%.2f' % (i/len(do.input_texts)*100))

i = 0
for target in do.target_texts:
	if len(target) > 220:
		i += 1
print(i, len(do.target_texts), '%.2f' % (i/len(do.target_texts)*100))



for txt in do.target_texts:
    print(txt)
print(len(do.target_texts))

import sys
sys.exit()
"""

# Converting tokens to integers (Neural Networks accept only integers as inputs), and
# reverse-lookup token index to decode sequences back to something readable.
input_token_index, target_token_index, reverse_input_token_index, reverse_target_token_index = \
    token_integer_mapping(do.input_tokens, do.target_tokens)

# Define input & output data and initialize them with zeros
encoder_input_data = np.zeros((n_input_samples, max_encoder_seq_length), dtype='int32')
decoder_input_data = np.zeros((n_input_samples, max_decoder_seq_length), dtype='int32')
decoder_target_data = np.zeros((n_input_samples, max_decoder_seq_length, num_decoder_tokens), dtype='float32')


# fill input data & one-hot encode targets
# Loop samples
for i, (input_list, target_list) in enumerate(zip(do.input_lists, do.target_lists)):
    # Loop input sequences
    for t, token in enumerate(input_list):
        encoder_input_data[i, t] = input_token_index[token]
    # Loop target sequences
    for t, token in enumerate(target_list):
        # decoder_target_data is ahead of decoder_input_data by one time step
        decoder_input_data[i, t] = target_token_index[token]
        if t > 0:
            # decoder_target_data will be ahead by one time step and will not include the start character. Initial value altered.
            decoder_target_data[i, t-1, target_token_index[token]] = 1.

# Build the encoder-decoder model
###encoder_inputs, decoder_inputs, decoder_outputs = build_model(latent_dim, num_encoder_tokens, num_decoder_tokens)
###model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model_name = 'c2p_att_plain_3072-26plus-03.hdf5'
model = load_model('checkpoints/'+model_name)
print("model name:", model_name)
# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# Save model after each epoch
checkpointer = ModelCheckpoint(filepath= 'checkpoints/c2p_att_plain_3072-29plus-{epoch:02d}.hdf5', verbose=1)
model.summary()
print("Training started at:", datetime.datetime.now())
print("================")
# Note that 'decoder_target_data' needs to be one-hot encoded, rather than sequences of integers like 'decoder_input_data'!
###model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs,
   ###       callbacks=[checkpointer], validation_split=0.2)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, callbacks=[checkpointer])
# Save model
model.save('c2p_att_plain_3072.h5')
###model.save('/home/aa043/sea/trained_models/code2pseudo/reduced_surface/c2p_att_redsurf.h5')
print("================")
print("Training completed at:", datetime.datetime.now())

