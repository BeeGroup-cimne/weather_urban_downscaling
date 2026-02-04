#!/usr/bin/env python3
"""
Transformer Optimizado para Weather Downscaling con Memory Management
Incluye optimizaciones específicas para reducir OOM en GPU training
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    TimeDistributed, Concatenate, LayerNormalization, 
    Dropout, Dense, Reshape, Permute, Activation
)
from tensorflow.keras.models import Model
from typing import Tuple

class OptimizedTransformerBlock(tf.keras.layers.Layer):
    """
    Transformer Block optimizado para memory efficiency
    Implementa atención con mecanismos de reducción de memoria
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, ff_dim: int = 512, 
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-head self attention
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'),  # GELU es más estable que ReLU
            Dropout(dropout_rate),
            Dense(embed_dim)
        ])
        
        # Normalization layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout_rate)
        
    def call(self, inputs, training=None):
        # Input shape: (batch, seq_len, embed_dim)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Self attention con memory optimization
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class MemoryEfficientTransformer:
    """
    Wrapper para Transformer con optimización de memoria
    Implementa chunking de atención para secuencias largas
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, 
                 chunk_size: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        
        self.transformer_blocks = []
        
    def add_transformer_block(self, **block_kwargs):
        """Añadir transformer block con configuración optimizada"""
        block = OptimizedTransformerBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            **block_kwargs
        )
        self.transformer_blocks.append(block)
        return block
    
    def apply_chunked_attention(self, x):
        """Aplicar atención por chunks para reducir memoria"""
        batch_size, seq_len, embed_dim = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        
        if seq_len <= self.chunk_size:
            # Sin chunking para secuencias cortas
            for block in self.transformer_blocks:
                x = block(x)
            return x
        
        # Chunking para secuencias largas
        outputs = []
        for i in range(0, seq_len, self.chunk_size):
            end_idx = tf.minimum(i + self.chunk_size, seq_len)
            chunk = x[:, i:end_idx, :]
            
            for block in self.transformer_blocks:
                chunk = block(chunk)
            
            outputs.append(chunk)
        
        return tf.concat(outputs, axis=1)

def build_optimized_transformer_unet(input_shape: Tuple[int, ...], 
                                   filters: int = 32,
                                   num_heads: int = 4,
                                   ff_dim: int = 256,
                                   num_transformer_blocks: int = 2,
                                   dropout_rate: float = 0.1,
                                   use_chunked_attention: bool = True) -> Model:
    """
    Construir Transformer U-Net optimizado para GPU server
    
    Args:
        input_shape: (seq_len, H, W, channels)
        filters: Filtros base para Conv layers
        num_heads: Número de cabezas de atención
        ff_dim: Dimensión feed-forward
        num_transformer_blocks: Número de bloques transformer
        dropout_rate: Tasa de dropout
        use_chunked_attention: Usar chunking para secuencias largas
    """
    
    seq_len, H, W, C = input_shape
    
    # --- ENCODER ---
    # Input separado para dinámico y estático (como en el código original)
    inp_dyn = Input(shape=(seq_len, H, W, 9), name='dynamic_input')
    inp_st = Input(shape=(seq_len, H, W, C-9), name='static_input')
    
    # Upsample LR a HR y concatenar con static
    x_up = TimeDistributed(
        tf.keras.layers.Resizing(H, W, interpolation="bilinear")
    )(inp_dyn)
    x = Concatenate()([x_up, inp_st])
    
    # Encoder path (igual que U-Net)
    c1 = TimeDistributed(Conv2D(filters, 3, activation='relu', padding='same'))(x)
    c1 = TimeDistributed(Conv2D(filters, 3, activation='relu', padding='same'))(c1)
    p1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
    
    c2 = TimeDistributed(Conv2D(filters*2, 3, activation='relu', padding='same'))(p1)
    c2 = TimeDistributed(Conv2D(filters*2, 3, activation='relu', padding='same'))(c2)
    p2 = TimeDistributed(MaxPooling2D((2, 2)))(c2)
    
    c3 = TimeDistributed(Conv2D(filters*4, 3, activation='relu', padding='same'))(p2)
    c3 = TimeDistributed(Conv2D(filters*4, 3, activation='relu', padding='same'))(c3)
    p3 = TimeDistributed(MaxPooling2D((2, 2)))(c3)
    
    # --- TRANSFORMER BOTTLENECK ---
    # Reducir dimensión espacial para transformer
    bottleneck_filters = filters * 8
    x_neck = TimeDistributed(
        Conv2D(bottleneck_filters, (1, 1), padding='same', activation='relu')
    )(p3)
    
    # Reshape para transformer: (batch, seq_len, embed_dim)
    # Usar dimensiones estáticas para evitar errores con KerasTensors
    h_reduced = H // 8
    w_reduced = W // 8
    embed_dim = h_reduced * w_reduced * bottleneck_filters
    x_flat = Reshape((seq_len, embed_dim))(x_neck)
    
    # Transformer blocks con memory optimization
    if use_chunked_attention:
        # Usar chunked attention para secuencias largas
        transformer = MemoryEfficientTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            chunk_size=min(512, embed_dim // num_heads)  # Chunk size adaptativo
        )
        
        for _ in range(num_transformer_blocks):
            transformer.add_transformer_block(
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            )
        
        x_trans = transformer.apply_chunked_attention(x_flat)
    else:
        # Transformer estándar
        x_trans = x_flat
        for _ in range(num_transformer_blocks):
            x_trans = OptimizedTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            )(x_trans)
    
    # Reshape de vuelta a formato espacial
    x_reshaped = Reshape((seq_len, h_reduced, w_reduced, bottleneck_filters))(x_trans)
    # Asegurar que los canales coincidan
    x_reshaped = TimeDistributed(
        Conv2D(bottleneck_filters, 1, padding='same', activation='relu')
    )(x_reshaped)
    
    # --- DECODER ---
    # Upsample y skip connections
    u3 = TimeDistributed(
        tf.keras.layers.Resizing(c3.shape[2], c3.shape[3], interpolation="bilinear")
    )(x_reshaped)
    u3 = Concatenate()([u3, c3])
    c4 = TimeDistributed(Conv2D(filters*4, 3, activation='relu', padding='same'))(u3)
    c4 = TimeDistributed(Conv2D(filters*4, 3, activation='relu', padding='same'))(c4)
    
    u2 = TimeDistributed(
        tf.keras.layers.Resizing(c2.shape[2], c2.shape[3], interpolation="bilinear")
    )(c4)
    u2 = Concatenate()([u2, c2])
    c5 = TimeDistributed(Conv2D(filters*2, 3, activation='relu', padding='same'))(u2)
    c5 = TimeDistributed(Conv2D(filters*2, 3, activation='relu', padding='same'))(c5)
    
    u1 = TimeDistributed(
        tf.keras.layers.Resizing(c1.shape[2], c1.shape[3], interpolation="bilinear")
    )(c5)
    u1 = Concatenate()([u1, c1])
    c6 = TimeDistributed(Conv2D(filters, 3, activation='relu', padding='same'))(u1)
    c6 = TimeDistributed(Conv2D(filters, 3, activation='relu', padding='same'))(c6)
    
    # --- OUTPUT ---
    # Solo el último timestep es relevante para downscaling
    out = TimeDistributed(
        Conv2D(1, (1, 1), activation='linear'), name='output'
    )(c6)
    
    # Extraer último timestep
    final_out = out[:, -1, :, :, :]  # (batch, H, W, 1)
    
    model = Model(
        inputs=[inp_dyn, inp_st], 
        outputs=final_out, 
        name=f"OptimizedTransformerUNet_{num_transformer_blocks}blocks"
    )
    
    return model

def build_lightweight_transformer_unet(input_shape: Tuple[int, ...], 
                                   max_memory_gb: int = 8) -> Model:
    """
    Construir Transformer ultra-optimizado para GPUs con memoria limitada
    
    Args:
        input_shape: (seq_len, H, W, channels)
        max_memory_gb: Máxima memoria GPU disponible en GB
    """
    
    # Configuración adaptativa según memoria disponible
    if max_memory_gb <= 8:
        return build_optimized_transformer_unet(
            input_shape=input_shape,
            filters=16,  # Reducido
            num_heads=2,  # Reducido
            ff_dim=128,   # Reducido
            num_transformer_blocks=1,  # Solo 1 bloque
            dropout_rate=0.2,
            use_chunked_attention=True
        )
    elif max_memory_gb <= 16:
        return build_optimized_transformer_unet(
            input_shape=input_shape,
            filters=24,
            num_heads=3,
            ff_dim=192,
            num_transformer_blocks=2,
            dropout_rate=0.15,
            use_chunked_attention=True
        )
    else:  # 24GB+
        return build_optimized_transformer_unet(
            input_shape=input_shape,
            filters=32,
            num_heads=4,
            ff_dim=256,
            num_transformer_blocks=3,
            dropout_rate=0.1,
            use_chunked_attention=False  # Memoria suficiente para estándar
        )

# Wrapper para compatibilidad con código existente
def build_transformer_optimized(input_shape, **kwargs):
    """Wrapper para compatibilidad con ModelZoo"""
    return build_optimized_transformer_unet(input_shape, **kwargs)

if __name__ == "__main__":
    # Test del modelo
    from config.gpu_server_config import GPUServerConfig as Config
    
    input_shape = (
        Config.SEQ_LEN,
        *Config.HR_SHAPE,
        Config.CHANNELS + Config.STATIC_CHANNELS
    )
    
    print(f"🔧 Testing Optimized Transformer UNet")
    print(f"   Input shape: {input_shape}")
    
    # Construir modelo
    model = build_lightweight_transformer_unet(
        input_shape=input_shape,
        max_memory_gb=Config.GPU_MEMORY_GB or 8
    )
    
    # Mostrar resumen
    print(f"   Model params: {model.count_params() / 1e6:.2f}M")
    
    # Test forward pass
    batch_size = Config.BATCH_SIZE
    dummy_input = [
        tf.random.normal((batch_size, Config.SEQ_LEN, *Config.HR_SHAPE, Config.CHANNELS)),
        tf.random.normal((batch_size, Config.SEQ_LEN, *Config.HR_SHAPE, Config.STATIC_CHANNELS))
    ]
    
    output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    print(f"   Memory per batch: ~{model.count_params() * batch_size * 4 / 1024**2:.2f}MB")
    
    print("✅ Transformer Optimizado funcionando correctamente!")
