import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def create_source_embeddings(data, d_model):
    # Assuming data is a list of unique source ids
    num_sources = len(data)
    source_emb = tf.Variable(initial_value=tf.random.normal((num_sources, d_model)), dtype=tf.float32, trainable=True, name="source_embeddings")

    # Create a mapping from source ids to indices in the source_emb tensor
    source_id_to_idx = {source_id: idx for idx, source_id in enumerate(data)}

    return source_emb, source_id_to_idx

def create_temporal_embeddings(data, d_model):
    # Assuming data is a list of timestamps
    num_timestamps = len(data)
    temporal_emb = tf.Variable(initial_value=tf.random.normal((num_timestamps, d_model)), dtype=tf.float32, trainable=True, name="temporal_embeddings")

    # Create a mapping from timestamps to indices in the temporal_emb tensor
    timestamp_to_idx = {timestamp: idx for idx, timestamp in enumerate(data)}

    return temporal_emb, timestamp_to_idx

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

# Define the necessary layers and components for the Iris model
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)
        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, source_embeddings):
        attn1, attn_weights_block1 = self.mha1(x, x, x, mask)  # self-attention
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1, source_embeddings, source_embeddings, None)  # source attention
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.mha3 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, source_embeddings):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # self-attention
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)  # encoder-decoder attention
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        attn3, attn_weights_block3 = self.mha3(out2, source_embeddings, source_embeddings, None)  # source attention
        attn3 = self.dropout3(attn3, training=training)
        out3 = self.layernorm3(attn3 + out2)

        ffn_output = self.ffn(out3)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out3)

        return out3, attn_weights_block1, attn_weights_block2, attn_weights_block3

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, max_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, source_embeddings, temporal_embeddings):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # Token embeddings
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]  # Positional embeddings
        x += source_embeddings  # Source embeddings
        x += temporal_embeddings  # Temporal embeddings
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2, block3 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask, source_embeddings)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
            attention_weights['decoder_layer{}_block3'.format(i + 1)] = block3

        return x, attention_weights

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_position_encoding, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, source_embeddings, temporal_embeddings):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # Token embeddings
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]  # Positional embeddings
        x += source_embeddings  # Source embeddings
        x += temporal_embeddings  # Temporal embeddings
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.enc_layers[i](x, training, mask, source_embeddings)
            attention_weights['encoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['encoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_position_encoding, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_position_encoding, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_position_encoding, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training, enc_padding_mask, look_ahead_mask, dec_padding_mask, source_embeddings, temporal_embeddings):
        enc_input, dec_input = inputs

        enc_output = self.encoder(enc_input, training, enc_padding_mask, source_embeddings, temporal_embeddings)
        dec_output, attention_weights = self.decoder(dec_input, enc_output, training, look_ahead_mask, dec_padding_mask, source_embeddings, temporal_embeddings)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights