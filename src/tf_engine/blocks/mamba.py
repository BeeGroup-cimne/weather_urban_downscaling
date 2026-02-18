import tensorflow as tf
from tensorflow.keras import layers

class SimpleMambaBlock(layers.Layer):
    """
    Implementación robusta de Mamba Block para Apple Silicon (M1/M2/M3/M4).
    Sustituye Conv1D (buggy en Metal) por DepthwiseConv2D (estable).
    """
    def __init__(self, model_dim, d_state=16, d_conv=4, expand=2, **kwargs):
        # Explicit super() avoids AutoGraph edge-cases (KeyError('__class__')).
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
