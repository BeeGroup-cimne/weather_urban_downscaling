from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, ConvLSTM2D,
    TimeDistributed, Concatenate, BatchNormalization, 
    Activation, Add, LeakyReLU, Resizing, Dropout, 
    MultiHeadAttention, Lambda, Permute, Dense, LayerNormalization, Reshape
)
from tensorflow.keras.models import Model
import tensorflow as tf
from config.config import Config
from .blocks import conv_block, res_block, SimpleMambaBlock, temporal_transformer_block

class DownsrUNet:
    def __init__(self, input_shape_dyn=None, input_shape_st=None, bottleneck_type='mamba'):
        self.input_shape_dyn = input_shape_dyn if input_shape_dyn else (Config.SEQ_LEN, *Config.LR_SHAPE, 9)
        self.input_shape_st = input_shape_st if input_shape_st else (Config.SEQ_LEN, *Config.HR_SHAPE, Config.STATIC_CHANNELS)
        self.bottleneck_type = bottleneck_type
        self.learning_rate = Config.LEARNING_RATE

    def get_optimizer(self, lr):
        """Selecciona el optimizador adecuado según el Hardware"""
        if Config.IS_MAC_SILICON:
            from tensorflow.keras.optimizers.legacy import Adam
            print("🚀 Using Legacy Adam (Metal/M-Series)")
            return Adam(learning_rate=lr)
        else:
            from tensorflow.keras.optimizers import Adam
            print("⚙️ Using Standard Adam")
            return Adam(learning_rate=lr)

    def build(self):
        """Construye el modelo según el tipo de bottleneck configurado"""
        inp_dyn = Input(shape=self.input_shape_dyn, name="lr_input")
        inp_st = Input(shape=self.input_shape_st, name="static_input")

        # --- 1. BRIDGE & FUSION ---
        # Escalamos la entrada LR al tamaño HR para concatenarla con los datos estáticos
        x_up = TimeDistributed(Resizing(self.input_shape_st[1], self.input_shape_st[2], interpolation="bilinear"))(inp_dyn)
        x = Concatenate(axis=-1)([x_up, inp_st])

        # --- 2. ENCODER (Espacial / Frame a Frame) ---
        skips = []
        
        # Block 1
        c1 = conv_block(x, 32)
        skips.append(c1)
        p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
        
        # Block 2
        c2 = conv_block(p1, 64)
        skips.append(c2)
        p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
        
        # Block 3
        c3 = conv_block(p2, 128)
        skips.append(c3)
        p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

        # --- 3. BOTTLENECK ---
        if self.bottleneck_type == 'mamba':
            x_neck = self._build_mamba_bottleneck(p3)
        elif self.bottleneck_type == 'lstm':
            x_neck = self._build_lstm_bottleneck(p3)
        elif self.bottleneck_type == 'transformer':
            x_neck = self._build_transformer_bottleneck(p3)
        else:
            # Default to standard CNN bottleneck (UNet classic)
            x_neck = conv_block(p3, 256)
            x_neck = TimeDistributed(Dropout(0.3))(x_neck)

        # --- 4. DECODER (Reconstrucción Espacial) ---
        # Usamos las Skip Connections (c3, c2, c1) -> skips[2], skips[1], skips[0]
        
        # Upsample 3
        u3 = TimeDistributed(Resizing(c3.shape[2], c3.shape[3]))(x_neck)
        u3 = Concatenate()([u3, c3])
        c4 = conv_block(u3, 128)

        # Upsample 2
        u2 = TimeDistributed(Resizing(c2.shape[2], c2.shape[3]))(c4)
        u2 = Concatenate()([u2, c2])
        c5 = conv_block(u2, 64)

        # Upsample 1
        u1 = TimeDistributed(Resizing(c1.shape[2], c1.shape[3]))(c5)
        u1 = Concatenate()([u1, c1])
        c6 = conv_block(u1, 32)

        # --- 5. OUTPUT ---
        out = TimeDistributed(Conv2D(1, (1, 1), activation='linear'))(c6)
        
        model_name = f"DownsrUNet_{self.bottleneck_type}"
        model = Model([inp_dyn, inp_st], out, name=model_name)
        model.compile(optimizer=self.get_optimizer(self.learning_rate), loss='mse', metrics=['mae'])
        
        return model

    def _build_mamba_bottleneck(self, x):
        # Shape actual: (Batch, Time, H', W', C)
        b_shape = tf.shape(x)
        h_enc, w_enc, c_enc = x.shape[2], x.shape[3], x.shape[4]
        
        # Aplanamos: (Batch, Sequence_Long, Channels)
        # Sequence_Long = Time * H * W
        x_flat = Reshape((-1, c_enc))(x) 
        
        # Aplicamos Bloques Mamba
        x_mamba = SimpleMambaBlock(model_dim=c_enc, expand=2)(x_flat)
        x_mamba = SimpleMambaBlock(model_dim=c_enc, expand=2)(x_mamba)
        
        # Des-aplanamos al formato espacial
        x_out = Reshape((Config.SEQ_LEN, h_enc, w_enc, c_enc))(x_mamba)
        return x_out

    def _build_lstm_bottleneck(self, x):
        # ConvLSTM procesa la secuencia temporal en el espacio latente
        lstm_out = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding="same", return_sequences=True)(x)
        lstm_out = TimeDistributed(BatchNormalization())(lstm_out)
        lstm_out = TimeDistributed(LeakyReLU(0.1))(lstm_out)
        return lstm_out

    def _build_transformer_bottleneck(self, x):
        x_neck = TimeDistributed(Conv2D(256, (1, 1), padding="same"))(x)
        x_trans = temporal_transformer_block(x_neck, embed_dim=256, num_heads=4)
        return x_trans
