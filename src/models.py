import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, ConvLSTM2D,
    TimeDistributed, Concatenate, BatchNormalization, 
    Activation, Add, LeakyReLU, Resizing, Dropout, 
    MultiHeadAttention, Lambda, Permute, Dense, LayerNormalization
)
from tensorflow.keras.models import Model
from config.config import Config


# ==========================================
# 3. MODELOS (Arquitecturas)
# ==========================================
class ModelZoo:
    @staticmethod
    def get_optimizer(lr):
        """Selecciona el optimizador adecuado según el Hardware"""
        if Config.IS_MAC_SILICON:
            from tensorflow.keras.optimizers.legacy import Adam
            print("🚀 Using Legacy Adam (Metal/M-Series)")
            return Adam(learning_rate=lr)
        else:
            from tensorflow.keras.optimizers import Adam
            print("⚙️ Using Standard Adam")
            return Adam(learning_rate=lr)

    @staticmethod
    def res_block(x, filters):
        skip = x
        x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
        x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = Add()([x, skip])
        return x

    @staticmethod
    def conv_block(x, filters):
        x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(LeakyReLU(0.1))(x)
        x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(LeakyReLU(0.1))(x)
        return x

    @staticmethod
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

    @classmethod
    def build_convlstm(cls):
        """Experimento 1: ConvLSTM + Upsampling"""
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, 9))
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, 18))

        x = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True)(inp_dyn)

        # Upsampling progresivo
        for _ in range(6):
            x = TimeDistributed(UpSampling2D((2, 2), interpolation='bilinear'))(x)
            x = TimeDistributed(Conv2D(64, (3, 3), padding="same"))(x)
            x = TimeDistributed(LeakyReLU(0.2))(x)

        x = TimeDistributed(Resizing(*Config.HR_SHAPE))(x)
        merged = Concatenate()([x, inp_st])

        x = TimeDistributed(Conv2D(64, (3, 3), padding="same"))(merged)
        x = cls.res_block(x, 64)
        out = TimeDistributed(Conv2D(1, (1, 1), activation="linear"))(x)

        model = Model([inp_dyn, inp_st], out, name="Exp1_ConvLSTM")
        model.compile(optimizer=cls.get_optimizer(Config.LEARNING_RATE), loss='mse', metrics=['mae'])
        return model

    @classmethod
    def build_unet(cls):
        """Experimento 2: U-Net Standard"""
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, 9))
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, 18))

        # Bridge
        x_up = TimeDistributed(Resizing(*Config.HR_SHAPE, interpolation="bilinear"))(inp_dyn)
        x = Concatenate()([x_up, inp_st])

        # Encoder
        c1 = cls.conv_block(x, 32)
        p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
        c2 = cls.conv_block(p1, 64)
        p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
        c3 = cls.conv_block(p2, 128)
        p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

        # Bottleneck
        b = cls.conv_block(p3, 256)
        b = TimeDistributed(Dropout(0.3))(b)

        # Decoder
        u3 = TimeDistributed(Resizing(c3.shape[2], c3.shape[3]))(b)
        u3 = Concatenate()([u3, c3])
        c4 = cls.conv_block(u3, 128)

        u2 = TimeDistributed(Resizing(c2.shape[2], c2.shape[3]))(c4)
        u2 = Concatenate()([u2, c2])
        c5 = cls.conv_block(u2, 64)

        u1 = TimeDistributed(Resizing(c1.shape[2], c1.shape[3]))(c5)
        u1 = Concatenate()([u1, c1])
        c6 = cls.conv_block(u1, 32)

        out = TimeDistributed(Conv2D(1, (1, 1), activation='linear'))(c6)

        model = Model([inp_dyn, inp_st], out, name="Exp2_UNet")
        model.compile(optimizer=cls.get_optimizer(Config.LEARNING_RATE), loss='mse', metrics=['mae'])
        return model

    @classmethod
    def build_hybrid_unet_lstm(cls):
        """
        ARQUITECTURA HÍBRIDA: U-Net + ConvLSTM
        --------------------------------------
        1. Encoder Espacial (TimeDistributed Conv2D): Comprime cada frame de la secuencia.
        2. Bottleneck Temporal (ConvLSTM): Aprende la evolución temporal en el espacio latente.
        3. Decoder Espacial (TimeDistributed UpSampling): Reconstruye la alta resolución.
        """
        # Inputs Dinámicos (LR) y Estáticos (HR)
        # Nota: Usamos '9' canales o 'None' si queremos flexibilidad total
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, 9)) 
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, 18))

        # --- 1. BRIDGE & FUSION ---
        # Escalamos la entrada LR al tamaño HR para concatenarla con los datos estáticos
        # Esto permite que la red vea la topografía (HR) desde la primera capa.
        x_up = TimeDistributed(Resizing(*Config.HR_SHAPE, interpolation="bilinear"))(inp_dyn)
        x = Concatenate()([x_up, inp_st])

        # --- 2. ENCODER (Espacial / Frame a Frame) ---
        # Usamos TimeDistributed para aplicar las mismas Convoluciones a cada paso de tiempo t
        
        # Bloque 1
        c1 = cls.conv_block(x, 32)
        p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
        
        # Bloque 2
        c2 = cls.conv_block(p1, 64)
        p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
        
        # Bloque 3
        c3 = cls.conv_block(p2, 128)
        p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

        # --- 3. BOTTLENECK (Temporal / ConvLSTM) ---
        # Aquí sustituimos la convolución normal por una ConvLSTM.
        # Esto procesa la secuencia temporal en el espacio latente (comprimido).
        # return_sequences=True es vital para mantener la dimensión de tiempo para el decoder.
        
        lstm_out = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding="same", return_sequences=True)(p3)
        lstm_out = TimeDistributed(BatchNormalization())(lstm_out)
        lstm_out = TimeDistributed(LeakyReLU(0.1))(lstm_out)
        
        # Podemos añadir una segunda capa LSTM si hay memoria suficiente (Opcional)
        # lstm_out = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding="same", return_sequences=True)(lstm_out)
        # lstm_out = TimeDistributed(BatchNormalization())(lstm_out)

        # --- 4. DECODER (Reconstrucción Espacial) ---
        # Usamos las Skip Connections (c3, c2, c1) del Encoder original
        
        # Upsample 1
        u3 = TimeDistributed(Resizing(c3.shape[2], c3.shape[3]))(lstm_out)
        u3 = Concatenate()([u3, c3]) # Skip Connection
        c4 = cls.conv_block(u3, 128)

        # Upsample 2
        u2 = TimeDistributed(Resizing(c2.shape[2], c2.shape[3]))(c4)
        u2 = Concatenate()([u2, c2]) # Skip Connection
        c5 = cls.conv_block(u2, 64)

        # Upsample 3
        u1 = TimeDistributed(Resizing(c1.shape[2], c1.shape[3]))(c5)
        u1 = Concatenate()([u1, c1]) # Skip Connection
        c6 = cls.conv_block(u1, 32)

        # --- 5. OUTPUT ---
        # Conv 1x1 para colapsar canales a 1 (Temperatura HR)
        out = TimeDistributed(Conv2D(1, (1, 1), activation='linear'))(c6)

        model = Model([inp_dyn, inp_st], out, name="Exp4_Hybrid_UNet_LSTM")
        model.compile(optimizer=cls.get_optimizer(Config.LEARNING_RATE), loss='mse', metrics=['mae'])
        return model    

    @classmethod
    def build_transformer(cls):
        """Experimento 3: U-Net con Transformer Bottleneck"""
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, 9))
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, 18))

        # Bridge & Encoder (Igual a U-Net)
        x_up = TimeDistributed(Resizing(*Config.HR_SHAPE, interpolation="bilinear"))(inp_dyn)
        x = Concatenate()([x_up, inp_st])

        c1 = cls.conv_block(x, 32)
        p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
        c2 = cls.conv_block(p1, 64)
        p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
        c3 = cls.conv_block(p2, 128)
        p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

        # --- TRANSFORMER BOTTLENECK ---
        x_neck = TimeDistributed(Conv2D(256, (1, 1), padding="same"))(p3)
        x_trans = cls.temporal_transformer_block(x_neck, embed_dim=256, num_heads=4)

        # Decoder
        u3 = TimeDistributed(Resizing(c3.shape[2], c3.shape[3]))(x_trans)
        u3 = Concatenate()([u3, c3])
        c4 = cls.conv_block(u3, 128)

        u2 = TimeDistributed(Resizing(c2.shape[2], c2.shape[3]))(c4)
        u2 = Concatenate()([u2, c2])
        c5 = cls.conv_block(u2, 64)

        u1 = TimeDistributed(Resizing(c1.shape[2], c1.shape[3]))(c5)
        u1 = Concatenate()([u1, c1])
        c6 = cls.conv_block(u1, 32)

        out = TimeDistributed(Conv2D(1, (1, 1), activation='linear'))(c6)

        model = Model([inp_dyn, inp_st], out, name="Exp3_TransformerUNet")
        model.compile(optimizer=cls.get_optimizer(Config.LEARNING_RATE), loss='mse', metrics=['mae'])
        return model
        
