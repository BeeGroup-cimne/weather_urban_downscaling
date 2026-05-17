import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, ConvLSTM2D,
    TimeDistributed, Concatenate, BatchNormalization, 
    Activation, Add, LeakyReLU, Resizing, Dropout, 
    MultiHeadAttention, Lambda, Permute, Dense, LayerNormalization
)
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from config.runtime import Config

# --- CLASE AUXILIAR: MAMBA BLOCK (Pure TF) ---

class SimpleMambaBlock(layers.Layer):
    """
    Implementación robusta de Mamba Block para Apple Silicon (M1/M2/M3/M4).
    Sustituye Conv1D (buggy en Metal) por DepthwiseConv2D (estable).
    """
    def __init__(self, model_dim, d_state=16, d_conv=4, expand=2, **kwargs):
        # Use explicit super() to avoid AutoGraph edge-cases where implicit super()
        # fails with KeyError('__class__') when traced.
        super(SimpleMambaBlock, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.d_inner = int(expand * model_dim)
        self.d_conv = d_conv
        self.d_state = d_state

        # Proyecciones de entrada
        self.in_proj = layers.Dense(self.d_inner * 2, use_bias=False)
        
        # --- 🛡️ FIX MAC SILICON 🛡️ ---
        # Usamos DepthwiseConv2D en lugar de Conv1D.
        # Esto evita el error "could not find registered platform" en Metal.
        self.dw_conv2d = layers.DepthwiseConv2D(
            kernel_size=(d_conv, 1), # Convolución sobre el eje de Tiempo (d_conv) x 1
            padding='same',          # 'same' mantiene la dimensión temporal estable
            depth_multiplier=1,
            data_format='channels_last'
        )
        # -----------------------------
        
        # Activación y Gating
        self.activation = layers.Activation('swish') 
        self.out_proj = layers.Dense(model_dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        # x shape: (Batch, Seq_Len, Channels)
        skip = x
        x = self.norm(x)
        
        # 1. Proyección y Split
        projected = self.in_proj(x)
        x_branch, z_branch = tf.split(projected, num_or_size_splits=2, axis=-1)
        
        # 2. Procesamiento Rama X (Simulando 1D con 2D)
        
        # --- 🔄 TRUCO DE RESHAPE 🔄 ---
        # Añadimos una dimensión falsa para que parezca una imagen 2D de ancho 1
        # Shape entra: (Batch, Time, Chan) -> Sale: (Batch, Time, 1, Chan)
        x_branch = tf.expand_dims(x_branch, axis=2)
        
        # Aplicamos la Convolución Robusta (GPU Friendly)
        x_branch = self.dw_conv2d(x_branch)
        
        # Quitamos la dimensión falsa
        # Shape entra: (Batch, Time, 1, Chan) -> Sale: (Batch, Time, Chan)
        x_branch = tf.squeeze(x_branch, axis=2)
        # ------------------------------
        
        x_branch = self.activation(x_branch)
        
        # 3. Gating
        z_branch = self.activation(z_branch)
        x_out = x_branch * z_branch
        
        # 4. Proyección de salida
        return self.out_proj(x_out) + skip

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_dim": self.model_dim,
            "d_state": self.d_state,
            "d_conv": self.d_conv
        })
        return config
        
# MODELOS (Arquitecturas)

class ModelZoo:
    @staticmethod
    def _resolve_lr_var_index(default_var="t2m", default_idx=3):
        """
        Resolve LR channel index for baselines.
        Priority:
          1) BASELINE_LR_CHANNEL env (int)
          2) BASELINE_LR_VAR env (name, resolved via stats_config.npz if available)
          3) default_var via stats_config.npz
          4) default_idx fallback
        """
        # 1) explicit channel
        env_ch = os.getenv("BASELINE_LR_CHANNEL", "").strip()
        if env_ch:
            try:
                return int(env_ch), f"channel_{int(env_ch)}"
            except Exception:
                pass

        def _try_resolve_by_name(var_name: str):
            if not var_name:
                return None
            var_name = str(var_name).strip().lower()
            stats_path = getattr(Config, "STATS_PATH", None)
            if not stats_path:
                return None
            try:
                import numpy as np
                if not os.path.exists(stats_path):
                    return None
                stats = np.load(stats_path, allow_pickle=True)
                if "lr_var_names" not in stats:
                    return None
                names = [str(v) for v in np.asarray(stats["lr_var_names"]).tolist()]
                names_low = [n.lower() for n in names]
                if var_name in names_low:
                    idx = names_low.index(var_name)
                    return idx, names[idx]
                partial = [i for i, n in enumerate(names_low) if var_name in n]
                if partial:
                    idx = partial[0]
                    return idx, names[idx]
            except Exception:
                return None
            return None

        # 2) env var name
        env_name = os.getenv("BASELINE_LR_VAR", "").strip()
        resolved = _try_resolve_by_name(env_name) if env_name else None
        if resolved is not None:
            return resolved

        # 3) default name
        resolved = _try_resolve_by_name(default_var)
        if resolved is not None:
            return resolved

        # 4) fallback index
        return int(default_idx), f"channel_{int(default_idx)}"

    @classmethod
    def build_lr_upsample_baseline(cls, interpolation="nearest"):
        """
        Baseline: upsample LR -> HR without learning.

        Notes:
        - Keeps the same input signature as other models: (x_lr, x_static).
        - Selects LR variable by name (default 't2m') when stats_config.npz is available,
          else falls back to channel index 3.
        - Static input is connected as a zero-tensor so the graph is not disconnected,
          but it does not affect the output.
        """
        from config.runtime import Config
        lr_c = int(getattr(Config, "CHANNELS", 9))
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, lr_c), name="lr_input")
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, Config.STATIC_CHANNELS), name="static_input")

        lr_idx, lr_name = cls._resolve_lr_var_index(default_var="t2m", default_idx=3)
        lr_idx = max(0, min(lr_idx, lr_c - 1))

        x = layers.Lambda(lambda z, i=lr_idx: z[..., i:i + 1], name=f"select_lr_{lr_name}")(inp_dyn)
        x_up = TimeDistributed(
            Resizing(*Config.HR_SHAPE, interpolation=interpolation),
            name=f"upsample_{interpolation}",
        )(x)

        # Connect static input without using it (avoid disconnected graph)
        st_zero = layers.Lambda(lambda s: tf.zeros_like(s[..., :1]), name="static_zero")(inp_st)
        out = Add(name="out")([x_up, st_zero])

        model = Model([inp_dyn, inp_st], out, name=f"Baseline_Upsample_{interpolation}_{lr_name}")
        model.compile(optimizer=cls.get_optimizer(Config.LEARNING_RATE), loss="mse", metrics=["mae", "mse"])
        return model

    @classmethod
    def build_lr_upsample_nearest(cls):
        return cls.build_lr_upsample_baseline(interpolation="nearest")

    @classmethod
    def build_lr_upsample_bilinear(cls):
        return cls.build_lr_upsample_baseline(interpolation="bilinear")

    @staticmethod
    def _l2():
        wd = getattr(Config, "L2_WEIGHT_DECAY", 0.0)
        return regularizers.l2(wd) if wd and wd > 0 else None

    @staticmethod
    def _maybe_gaussian_noise(x):
        std = getattr(Config, "GAUSSIAN_NOISE_STD", 0.0)
        if std and std > 0:
            return layers.GaussianNoise(std)(x)
        return x

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
        reg = ModelZoo._l2()
        x = TimeDistributed(Conv2D(filters, (3, 3), padding="same", kernel_regularizer=reg))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(LeakyReLU(0.1))(x)
        x = TimeDistributed(Conv2D(filters, (3, 3), padding="same", kernel_regularizer=reg))(x)
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
        # key_dim is per-head dimensionality. Using embed_dim directly here explodes memory.
        head_dim = max(16, embed_dim // max(1, num_heads))
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(x_norm, x_norm)
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


    @staticmethod
    def build_convlstm(lr_shape=(5, 9), hr_shape=(251, 251)):
        """
        ConvLSTM: Versión Optimizada.
        CAMBIO: Usamos LayerNormalization en lugar de BatchNormalization.
        - Más rápido (sin TimeDistributed).
        - Nativo para 5D (sin errores de dimensión).
        - Mejor convergencia para LSTMs.
        """
        from config.runtime import Config
        
        # 1. Inputs
        lr_input = Input(shape=(Config.SEQ_LEN, lr_shape[0], lr_shape[1], Config.CHANNELS), name="lr_input")
        static_input = Input(shape=(Config.SEQ_LEN, hr_shape[0], hr_shape[1], Config.STATIC_CHANNELS), name="static_input")

        # 2. Upsampling
        scale_h = hr_shape[0] // lr_shape[0]
        scale_w = hr_shape[1] // lr_shape[1]
        
        x_lr_up = layers.TimeDistributed(
            layers.UpSampling2D(size=(scale_h, scale_w), interpolation='bilinear')
        )(lr_input)
        x_lr_up = layers.TimeDistributed(layers.Resizing(hr_shape[0], hr_shape[1]))(x_lr_up)

        # 3. Fusión
        x = layers.Concatenate(axis=-1)([x_lr_up, static_input])

        # 4. Procesamiento ConvLSTM
        # Block 1
        x = layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True, activation='tanh')(x)
        x = layers.LayerNormalization(axis=-1)(x) # <-- Rápido y soporta 5D
        
        # Block 2
        x = layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True, activation='tanh')(x)
        x = layers.LayerNormalization(axis=-1)(x) # <-- Rápido y soporta 5D
        
        # 5. Salida
        output = layers.TimeDistributed(layers.Conv2D(1, (1, 1), activation='linear'))(x)

        return Model(inputs=[lr_input, static_input], outputs=output, name="ConvLSTM")

    @classmethod
    def build_unet(cls):
        """Experimento 2: U-Net Standard"""
        from config.runtime import Config
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, 9))
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, Config.STATIC_CHANNELS))

        # Bridge
        x_dyn = cls._maybe_gaussian_noise(inp_dyn)
        x_up = TimeDistributed(Resizing(*Config.HR_SHAPE, interpolation="bilinear"))(x_dyn)
        x = Concatenate()([x_up, inp_st])

        # Encoder
        base = getattr(Config, "UNET_BASE_FILTERS", 32)
        c1 = cls.conv_block(x, base)
        p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
        c2 = cls.conv_block(p1, base * 2)
        p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
        c3 = cls.conv_block(p2, base * 4)
        p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)

        # Bottleneck
        b = cls.conv_block(p3, base * 8)
        b = TimeDistributed(Dropout(0.3))(b)

        # Decoder
        u3 = TimeDistributed(Resizing(c3.shape[2], c3.shape[3]))(b)
        u3 = Concatenate()([u3, c3])
        c4 = cls.conv_block(u3, base * 4)

        u2 = TimeDistributed(Resizing(c2.shape[2], c2.shape[3]))(c4)
        u2 = Concatenate()([u2, c2])
        c5 = cls.conv_block(u2, base * 2)

        u1 = TimeDistributed(Resizing(c1.shape[2], c1.shape[3]))(c5)
        u1 = Concatenate()([u1, c1])
        c6 = cls.conv_block(u1, base)

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
        from config.runtime import Config
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, 9)) 
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, Config.STATIC_CHANNELS))

        # --- 1. BRIDGE & FUSION ---
        # Escalamos la entrada LR al tamaño HR para concatenarla con los datos estáticos
        # Esto permite que la red vea la topografía (HR) desde la primera capa.
        x_dyn = cls._maybe_gaussian_noise(inp_dyn)
        x_up = TimeDistributed(Resizing(*Config.HR_SHAPE, interpolation="bilinear"))(x_dyn)
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

        model = Model([inp_dyn, inp_st], out, name="ConvLSTM_UNet")
        model.compile(optimizer=cls.get_optimizer(Config.LEARNING_RATE), loss='mse', metrics=['mae'])
        return model    

    @classmethod
    def build_transformer(cls):
        """Experimento 3: U-Net con Transformer Bottleneck"""
        inp_dyn = Input(shape=(Config.SEQ_LEN, *Config.LR_SHAPE, 9))
        inp_st = Input(shape=(Config.SEQ_LEN, *Config.HR_SHAPE, Config.STATIC_CHANNELS))

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
        transformer_dim = int(getattr(Config, "MODEL_DIM", 128))
        transformer_dim = max(64, min(transformer_dim, 256))
        x_neck = TimeDistributed(Conv2D(transformer_dim, (1, 1), padding="same"))(p3)
        x_trans = cls.temporal_transformer_block(
            x_neck,
            embed_dim=transformer_dim,
            num_heads=4,
            ff_dim=transformer_dim * 2,
        )

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

    @staticmethod
    def build_hybrid_unet_mamba(lr_shape=(5, 9), hr_shape=(251, 251)):
        """
        Arquitectura Híbrida:
        - Encoder: Conv2D (Extrae features espaciales)
        - Bottleneck: Mamba Block (Mezcla contexto global temporal-espacial)
        - Decoder: Conv2DTranspose (Reconstruye detalles)
        """
        from config.runtime import Config # Import local para evitar ciclos
        
        # 1. Inputs
        # LR Input: (Batch, Time, H_lr, W_lr, C_lr)
        lr_input = Input(shape=(Config.SEQ_LEN, lr_shape[0], lr_shape[1], Config.CHANNELS), name="lr_input")
        # Static Input: (Batch, Time, H_hr, W_hr, C_st)
        static_input = Input(shape=(Config.SEQ_LEN, hr_shape[0], hr_shape[1], Config.STATIC_CHANNELS), name="static_input")

        # 2. Pre-Procesado (Early Upsampling de LR)
        # Escalamos la imagen pequeña (5x9) al tamaño grande (251x251) para concatenar
        # Reescalar LR directamente al tamaño HR (robusto para cualquier tamaño)
        x_lr = ModelZoo._maybe_gaussian_noise(lr_input)
        x_lr_up = layers.TimeDistributed(
            layers.Resizing(hr_shape[0], hr_shape[1], interpolation='bilinear')
        )(x_lr)

        # 3. Concatenación (Early Fusion)
        x = layers.Concatenate(axis=-1)([x_lr_up, static_input])

        # --- ENCODER (TimeDistributed Conv2D) ---
        # Extraemos características frame a frame
        skips = []
        reg = ModelZoo._l2()
        
        # Block 1
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        skips.append(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x) # -> 125x125

        # Block 2
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        skips.append(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x) # -> 62x62

        # Block 3
        x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        skips.append(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x) # -> 31x31

        # --- MAMBA BOTTLENECK (The Core) ---
        # Aquí ocurre la magia. Aplanamos Espacio y Tiempo para que Mamba los mezcle.
        
        # Shape actual: (Batch, Time, H', W', C)
        b_shape = tf.shape(x)
        h_enc, w_enc, c_enc = x.shape[2], x.shape[3], x.shape[4]
        
        # Aplanamos: (Batch, Sequence_Long, Channels)
        # Sequence_Long = Time * H * W. Mamba verá todo el video como una tira larga.
        x_flat = layers.Reshape((-1, c_enc))(x) 
        
        # Aplicamos Bloques Mamba
        # Esto permite que un píxel en t=0 influya en un píxel en t=2 muy lejos espacialmente
        x_mamba = SimpleMambaBlock(model_dim=c_enc, expand=2)(x_flat)
        x_mamba = SimpleMambaBlock(model_dim=c_enc, expand=2)(x_mamba)
        
        # Des-aplanamos al formato espacial
        x = layers.Reshape((Config.SEQ_LEN, h_enc, w_enc, c_enc))(x_mamba)

        # --- DECODER (Reconstrucción) ---
        
        # Up 3
        x = layers.TimeDistributed(layers.UpSampling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        # Ajuste de shape por si el pooling fue impar
        x = layers.TimeDistributed(layers.Resizing(skips[2].shape[2], skips[2].shape[3]))(x)
        x = layers.Concatenate()([x, skips[2]])
        
        # Up 2
        x = layers.TimeDistributed(layers.UpSampling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        x = layers.TimeDistributed(layers.Resizing(skips[1].shape[2], skips[1].shape[3]))(x)
        x = layers.Concatenate()([x, skips[1]])
        
        # Up 1
        x = layers.TimeDistributed(layers.UpSampling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        x = layers.TimeDistributed(layers.Resizing(skips[0].shape[2], skips[0].shape[3]))(x)
        x = layers.Concatenate()([x, skips[0]])

        # Salida Final
        output = layers.TimeDistributed(layers.Conv2D(1, (1, 1), activation='linear'))(x)

        return Model(inputs=[lr_input, static_input], outputs=output, name="Hybrid_UNet_Mamba")

    @staticmethod
    def build_hybrid_unet_mamba_v2(lr_shape=(5, 9), hr_shape=(251, 251)):
        """
        Mamba UNet con Static Skip Connections directas al decoder.

        Diferencia vs v1: añade un pathway paralelo que procesa las features
        estáticas de morfología urbana (building_density, svf, etc.) a 3
        resoluciones y las inyecta en cada etapa del decoder, bypassing el
        bottleneck de Mamba. Esto garantiza un gradiente limpio desde la
        morfología urbana hasta la salida, solucionando el bajo R² (0.009)
        con building_density que tiene v1 frente a ConvLSTM (R²=0.178).

        Checkpoints de v1 no son compatibles: esta función genera un modelo
        nuevo con nombre "Hybrid_UNet_Mamba_v2".
        """
        from config.runtime import Config

        # --- Inputs ---
        lr_input = Input(
            shape=(Config.SEQ_LEN, lr_shape[0], lr_shape[1], Config.CHANNELS),
            name="lr_input"
        )
        static_input = Input(
            shape=(Config.SEQ_LEN, hr_shape[0], hr_shape[1], Config.STATIC_CHANNELS),
            name="static_input"
        )

        # --- Pre-procesado: upscale LR + early fusion (igual que v1) ---
        x_lr = ModelZoo._maybe_gaussian_noise(lr_input)
        x_lr_up = layers.TimeDistributed(
            layers.Resizing(hr_shape[0], hr_shape[1], interpolation='bilinear')
        )(x_lr)
        x = layers.Concatenate(axis=-1)([x_lr_up, static_input])

        reg = ModelZoo._l2()

        # --- STATIC PARALLEL PATHWAY ---
        # Pathway independiente que lleva las features morfológicas directamente
        # al decoder sin pasar por el aplanamiento 1D del bottleneck de Mamba.
        # Tres niveles de resolución para que coincidan con los skips del decoder.

        # Nivel 0: 251x251 → 16 canales
        st0 = layers.TimeDistributed(
            layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
            name="static_conv_s0"
        )(static_input)

        # Nivel 1: ~125x125 → 32 canales
        st1 = layers.TimeDistributed(layers.AveragePooling2D((2, 2)), name="static_pool_s1")(st0)
        st1 = layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
            name="static_conv_s1"
        )(st1)

        # Nivel 2: ~62x62 → 64 canales
        st2 = layers.TimeDistributed(layers.AveragePooling2D((2, 2)), name="static_pool_s2")(st1)
        st2 = layers.TimeDistributed(
            layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
            name="static_conv_s2"
        )(st2)

        # --- ENCODER (igual que v1) ---
        skips = []

        # Block 1 (251x251 → 125x125)
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        skips.append(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

        # Block 2 (125x125 → 62x62)
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        skips.append(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

        # Block 3 (62x62 → 31x31)
        x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        skips.append(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

        # --- MAMBA BOTTLENECK (igual que v1) ---
        h_enc, w_enc, c_enc = x.shape[2], x.shape[3], x.shape[4]

        x_flat = layers.Reshape((-1, c_enc))(x)
        x_mamba = SimpleMambaBlock(model_dim=c_enc, expand=2)(x_flat)
        x_mamba = SimpleMambaBlock(model_dim=c_enc, expand=2)(x_mamba)
        x = layers.Reshape((Config.SEQ_LEN, h_enc, w_enc, c_enc))(x_mamba)

        # --- DECODER con STATIC SKIP CONNECTIONS ---
        # En cada etapa del decoder concatenamos el static pathway correspondiente
        # junto con el skip del encoder. Esto da al modelo una ruta de gradiente
        # directa desde la morfología hasta la salida, sin pasar por Mamba.

        # Up 3: 31x31 → 62x62  [skip canal: 128 + 128 + 64 = 320]
        x = layers.TimeDistributed(layers.UpSampling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        x = layers.TimeDistributed(layers.Resizing(skips[2].shape[2], skips[2].shape[3]))(x)
        st2_r = layers.TimeDistributed(
            layers.Resizing(skips[2].shape[2], skips[2].shape[3]), name="st2_resize"
        )(st2)
        x = layers.Concatenate()([x, skips[2], st2_r])

        # Up 2: 62x62 → 125x125  [skip canal: 64 + 64 + 32 = 160]
        x = layers.TimeDistributed(layers.UpSampling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        x = layers.TimeDistributed(layers.Resizing(skips[1].shape[2], skips[1].shape[3]))(x)
        st1_r = layers.TimeDistributed(
            layers.Resizing(skips[1].shape[2], skips[1].shape[3]), name="st1_resize"
        )(st1)
        x = layers.Concatenate()([x, skips[1], st1_r])

        # Up 1: 125x125 → 251x251  [skip canal: 32 + 32 + 16 = 80]
        x = layers.TimeDistributed(layers.UpSampling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg))(x)
        x = layers.TimeDistributed(layers.Resizing(skips[0].shape[2], skips[0].shape[3]))(x)
        st0_r = layers.TimeDistributed(
            layers.Resizing(skips[0].shape[2], skips[0].shape[3]), name="st0_resize"
        )(st0)
        x = layers.Concatenate()([x, skips[0], st0_r])

        # Salida Final
        output = layers.TimeDistributed(layers.Conv2D(1, (1, 1), activation='linear'))(x)

        return Model(inputs=[lr_input, static_input], outputs=output, name="Hybrid_UNet_Mamba_v2")

