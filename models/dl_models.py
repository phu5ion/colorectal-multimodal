# LSTM
def create_masking_model(bidir=1):
    ts_input = Input((169, 1))
    kernel_regularizer=L1L2(l1=0.01, l2=0.01)
    # LSTM-input
    mask = tf.keras.layers.Masking(mask_value=0.0)(ts_input)
    if bidir:
        lstm_layer=Bidirectional(LSTM(8, return_sequences=True, dropout = 0.2, recurrent_dropout=0.2, activation='tanh', kernel_regularizer=kernel_regularizer, kernel_initializer="glorot_uniform"))(mask)
    else:
        lstm_layer=LSTM(16, return_sequences=True, dropout = 0.2, recurrent_dropout=0.2, activation='tanh', kernel_regularizer=kernel_regularizer, kernel_initializer="glorot_uniform")(mask)
    lstm_layer=LSTM(8, dropout = 0.2, activation='tanh', recurrent_dropout=0.2, kernel_regularizer=kernel_regularizer, kernel_initializer="glorot_uniform")(lstm_layer)
    output=Dense(1, activation='sigmoid')(lstm_layer)
    # define a model with a list of two inputs
    model = Model(inputs=ts_input, outputs=output)
    return model

# Temporal Convolutional Network (TCN aka Wavenet)
def create_tcn(num_channels, kernel_size=2, strides=1, dropout=0.1):
    # Initialise required stuff
    ts_input = Input(shape=(169, 1))
    kernel_regularizer=L1L2(l1=0.00, l2=0.00)
    kernel_initializer="he_uniform" # Instead of glorot_uniform as he_uniform seems to be theoretically better for relu and relu-like activations
    lnorm=LayerNormalization() # LayerNorm and BatchNorm both doesn't work...?
    # Depending on the number of levels, we increase dilation rate.
    # Number of levels should be self-calculated...
    num_levels = len(num_channels) # It should look like a list of length levels, how many filters each level
    inputs = ts_input
    for i in range(num_levels):
        dilation_size = 2 ** i
        out_channels = num_channels[i]
        
        # This is 1 block
        cnn1 = Conv1D(filters=out_channels, kernel_size=kernel_size, strides=strides, padding='causal', data_format='channels_last', activation='relu', dilation_rate=dilation_size, kernel_initializer=kernel_initializer)(inputs)
        #norm1 = lnorm(cnn1)
        dropout1 = Dropout(dropout, noise_shape=[tf.constant(1), tf.constant(1), tf.constant(out_channels)])(cnn1) # Noise_shape is to apply uniform dropout to all timesteps
        cnn2 = Conv1D(filters=out_channels, kernel_size=kernel_size, strides=strides, padding='causal', data_format='channels_last', activation='relu', dilation_rate=dilation_size, kernel_initializer=kernel_initializer)(dropout1)
        #norm2 = lnorm(cnn2)
        dropout2 = Dropout(dropout, noise_shape=[tf.constant(1), tf.constant(1), tf.constant(out_channels)])(cnn2) # Noise_shape is to apply uniform dropout to all timesteps
        out = relu(dropout2 + inputs) # Skip connections
        inputs = out
            
    out = out[:, -1, :]
    output = Dense(1, activation='sigmoid')(out)
    
    
    # define a model with a list of two inputs
    model = Model(inputs=ts_input, outputs=output)
    return model
