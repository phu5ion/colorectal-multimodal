# Code for positional encodings
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8, future_mask=None, use_cnn=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads # query, value and key vector dim will be embedding dim // num_heads
         
        if use_cnn:
            self.query_dense = Conv1D(embed_dim, kernel_size=3, padding="causal")
            self.key_dense = Conv1D(embed_dim, kernel_size=3, padding="causal")
            self.value_dense = Conv1D(embed_dim, kernel_size=3, padding="causal")
        else:
            self.query_dense = Dense(embed_dim)
            self.key_dense = Dense(embed_dim)
            self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)
        self.future_mask = future_mask
    def attention(self, query, key, value, mask=None):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        if mask is not None:
            scaled_score += (mask * -1e9)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # transposes along dimension 0 (the batch)

    def call(self, inputs): # Query, key and value are the input tensors to attention model
        # x.shape = [batch_size, seq_len, embedding_dim]
        query_input, key_input, value_input = inputs
        batch_size = tf.shape(query_input)[0]
        # Initialise query, key and value vectors
        query = self.query_dense(query_input)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(key_input)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(value_input)  # (batch_size, seq_len, embed_dim)
        # Separate out an extra dim by num_heads. So calculation is done separately for each head. 
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        # Perform the attention calculation
        attention, weights = self.attention(query, key, value, self.future_mask)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim) # Do the transpose to make it easier to combine the last two dimensions later on 
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim) # embed_dim = projection_dim * num_heads
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim) # do a linear projection
        return output
def transformer(d_model, stack=2, h=8, dropout=0.1, local_att_size=10, mask='local', use_cnn=True):
    # stack: num encoder and decoder stacks, each
    # h: num heads. h>1 for multi-headed attention
    kernel_init='he_uniform'
    
    ts_input = Input(shape=(169, 1))
    # Positional encodings
    pos_encodings = positional_encoding(ts_length, d_model)
    # Embedding
    embedding = Dense(d_model, activation='linear', name='original_encodings')(ts_input + pos_encodings) # Linear encoding # Dense layer with 1dim is actually the same as an Embedding layer
    # Create look_ahead_mask for causal attention. This works on the global sequence
    seq_len = 169
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0) # Set lower triangular part to 1, upper to 0
    # We also create a local mask that also implements causal attention so that it doesn't need to look so far behind
    local_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), local_att_size, 0)
    # We train this as a stack of attentional layers with causal masking
    x = embedding
    
    # Which mask to use?
    if mask=='local':
        mask = local_mask
    else:
        mask = look_ahead_mask
    # Encoder
    for i in range(stack):
        # This is 1 transformer block
        # Self attention
        residual = x
        x = MultiHeadSelfAttention(embed_dim=d_model, num_heads=h, future_mask=None, use_cnn=use_cnn)([x,x,x]) # Passing in list of query, key and value inputs for self-attention
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x+residual)
        # Feed-forward
        residual = x
        x = Sequential([Dense(d_model, activation="relu", kernel_initializer=kernel_init), Dense(d_model),])(x) # feed-forward layer with relu in between
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x+residual)
    enc_output = x # We save this in our memory
    # Decoder
    for i in range(stack):
        # This is 1 transformer block
        # Self attention
        residual = x
        x = MultiHeadSelfAttention(embed_dim=d_model, num_heads=h, future_mask=mask, use_cnn=use_cnn)([x,x,x]) 
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x+residual)
        # Encoder-decoder attention
        residual = x
        x = MultiHeadSelfAttention(embed_dim=d_model, num_heads=h, future_mask=mask, use_cnn=use_cnn)([x,enc_output,enc_output]) # key and value are our encoder output/memory
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x+residual)
        # Feed-forward
        residual = x
        x = Sequential([Dense(d_model, activation="relu", kernel_initializer=kernel_init), Dense(d_model),])(x) # feed-forward layer with relu in between
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x+residual)
        
    # Apply dense layer
    x = GlobalAveragePooling1D()(x)
    x = Dense(20, activation='relu', kernel_initializer=kernel_init)(x)
    outputs = Dense(1, activation="sigmoid", )(x)

    model = Model(inputs=ts_input, outputs=outputs)
    return model
