import tensorflow as tf
from tensorflow.keras.layers import Permute, Lambda, LayerNormalization, MultiHeadAttention, Dropout, Add, Dense

def temporal_transformer_block(x_input, embed_dim, num_heads=4, ff_dim=512):
    """Bloque Transformer para datos Espacio-Temporales"""
    # Shape: (Batch, Time, H, W, C)
    x = Permute((2, 3, 1, 4))(x_input)  # -> (Batch, H, W, Time, C)

    def flatten_spatial(x):
        s = tf.shape(x)
        return tf.reshape(x, (-1, s[3], s[4]))  # (Batch*H*W, Time, C)

    x_reshaped = Lambda(flatten_spatial)(x)
    x_norm = LayerNormalization(epsilon=1e-6)(x_reshaped)
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x_norm, x_norm)
    attn_out = Dropout(0.1)(attn_out)
    out1 = Add()([x_reshaped, attn_out])

    x_norm2 = LayerNormalization(epsilon=1e-6)(out1)
    ffn = Dense(ff_dim, activation="gelu")(x_norm2)
    ffn = Dense(embed_dim)(ffn)
    out2 = Add()([out1, ffn])

    def restore_spatial(args):
        x_proc, x_orig = args
        s = tf.shape(x_orig)  # (Batch, Time, H, W, C)
        return tf.reshape(x_proc, (s[0], s[2], s[3], s[1], s[4]))

    out_restored = Lambda(restore_spatial)([out2, x_input])
    return Permute((3, 1, 2, 4))(out_restored)
