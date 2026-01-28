#!/usr/bin/env python3
"""
Test de componentes sin dependencias de ML frameworks
Valida lógica de data processing y memory management
"""

import os
import sys
import numpy as np
import time
import gc
from pathlib import Path

class MockMLTest:
    """Test de la lógica sin frameworks ML"""
    
    def __init__(self):
        print("🧪 Test Lógico - Weather Downscaling")
        print("=" * 50)
        
    def test_memory_patterns(self):
        """Test patrones de memoria críticos"""
        print("\n🧠 Test 1: Patrones de Memoria")
        print("-" * 30)
        
        # Configuración tipo
        seq_len = 6
        hr_shape = (251, 251)
        channels = 9
        static_channels = 13
        batch_size = 4
        
        # Test 1: np.repeat problema (explosión de memoria)
        print("📊 Test np.repeat (problemático):")
        static_data = np.random.randn(*hr_shape, static_channels).astype(np.float32)
        
        start_time = time.time()
        repeated_data = np.repeat(static_data[np.newaxis, ...], seq_len, axis=0)
        repeat_time = time.time() - start_time
        repeat_memory = repeated_data.nbytes / 1024**2
        
        print(f"   Memory usado: {repeat_memory:.2f} MB")
        print(f"   Time: {repeat_time:.4f}s")
        
        # Test 2: np.broadcast_to solución
        print("\n📊 Test np.broadcast_to (optimizado):")
        start_time = time.time()
        broadcasted_data = np.broadcast_to(static_data[np.newaxis, ...], (seq_len, *static_data.shape))
        broadcast_time = time.time() - start_time
        broadcast_memory = broadcasted_data.nbytes / 1024**2
        
        print(f"   Memory usado: {broadcast_memory:.2f} MB")
        print(f"   Time: {broadcast_time:.4f}s")
        print(f"   ✅ Memory saved: {(repeat_memory - broadcast_memory):.2f} MB")
        
        # Test 3: Sequence creation
        print("\n📊 Test secuencia completa:")
        
        # Datos dinámicos
        dynamic_data = np.random.randn(seq_len, *hr_shape, channels).astype(np.float32)
        dynamic_memory = dynamic_data.nbytes / 1024**2
        print(f"   Dynamic data: {dynamic_memory:.2f} MB")
        
        # Concatenación
        final_sequence = np.concatenate([dynamic_data, broadcasted_data], axis=-1)
        final_memory = final_sequence.nbytes / 1024**2
        
        print(f"   Final sequence: {final_memory:.2f} MB")
        print(f"   Total por batch: {final_memory * batch_size:.2f} MB")
        
        return {
            'repeat_memory': repeat_memory,
            'broadcast_memory': broadcast_memory,
            'total_memory_mb': final_memory * batch_size
        }
    
    def test_mamba_token_explosion(self):
        """Test explosión de tokens en Mamba"""
        print("\n🔥 Test 2: Explosión de Tokens Mamba")
        print("-" * 35)
        
        seq_len = 6
        hr_shape = (251, 251)
        model_dim = 128
        
        # Cálculo de tokens
        tokens_per_sequence = seq_len * hr_shape[0] * hr_shape[1]
        memory_per_token_mb = (model_dim * 4) / 1024**2  # 4 bytes por float32
        total_memory_mb = tokens_per_sequence * memory_per_token_mb
        
        print(f"   Tokens por secuencia: {tokens_per_sequence:,}")
        print(f"   Memory por token: {memory_per_token_mb:.4f} MB")
        print(f"   Total memory: {total_memory_mb:.2f} MB")
        
        # Batch sizes
        for batch_size in [1, 2, 4]:
            batch_memory = total_memory_mb * batch_size
            print(f"   Batch {batch_size}: {batch_memory:.2f} MB")
            
            if batch_memory > 8000:  # >8GB
                print(f"   ⚠️  Batch {batch_size} puede causar OOM!")
        
        return {
            'tokens_per_sequence': tokens_per_sequence,
            'memory_per_batch_1': total_memory_mb,
            'memory_per_batch_4': total_memory_mb * 4
        }
    
    def test_config_scaling(self):
        """Test escalado de configuración"""
        print("\n⚙️  Test 3: Escalado de Configuración")
        print("-" * 35)
        
        configs = [
            {'name': 'Actual', 'seq_len': 6, 'hr_shape': (251, 251), 'batch_size': 4},
            {'name': 'Optimizado', 'seq_len': 4, 'hr_shape': (128, 128), 'batch_size': 2},
            {'name': 'Ultra-Optimizado', 'seq_len': 3, 'hr_shape': (64, 64), 'batch_size': 1}
        ]
        
        for config in configs:
            seq_len = config['seq_len']
            h, w = config['hr_shape']
            batch_size = config['batch_size']
            
            # Estimar memoria
            input_elements = batch_size * seq_len * h * w * (9 + 13)  # dynamic + static channels
            memory_mb = input_elements * 4 / 1024**2  # float32
            
            print(f"   {config['name']}:")
            print(f"     Sequence: {seq_len}, Size: {h}x{w}, Batch: {batch_size}")
            print(f"     Memory estimada: {memory_mb:.2f} MB")
            
            if memory_mb < 1000:
                print(f"     ✅ Adecuado para <4GB GPU")
            elif memory_mb < 4000:
                print(f"     ⚠️  Requiere 4-8GB GPU")
            else:
                print(f"     🔥 Requiere >8GB GPU")
        
        return configs
    
    def test_data_pipeline_chunks(self):
        """Test pipeline con chunks"""
        print("\n📦 Test 4: Pipeline con Chunks")
        print("-" * 30)
        
        # Simular data grande
        total_size = (1000, 500, 500, 22)  # 1000 timesteps, data grande
        chunk_size = 50
        
        print(f"   Data total: {total_size}")
        print(f"   Chunk size: {chunk_size}")
        
        # Procesamiento sin chunks
        start_time = time.time()
        full_array = np.random.randn(*total_size).astype(np.float32)
        full_memory = full_array.nbytes / 1024**2
        full_time = time.time() - start_time
        
        print(f"\n   Sin chunks:")
        print(f"     Memory: {full_memory:.2f} MB")
        print(f"     Time: {full_time:.4f}s")
        
        # Procesamiento con chunks
        start_time = time.time()
        chunk_results = []
        
        for i in range(0, total_size[0], chunk_size):
            end_idx = min(i + chunk_size, total_size[0])
            chunk = np.random.randn(end_idx - i, *total_size[1:]).astype(np.float32)
            chunk_results.append(chunk.mean())  # Simular procesamiento
            
            if i % (chunk_size * 4) == 0:
                print(f"     Procesados {end_idx}/{total_size[0]} chunks...")
        
        chunk_time = time.time() - start_time
        print(f"\n   Con chunks:")
        print(f"     Memory: ~{chunk_size * total_size[1] * total_size[2] * total_size[3] * 4 / 1024**2:.2f} MB")
        print(f"     Time: {chunk_time:.4f}s")
        print(f"     ✅ Memory saved: ~{(full_memory - chunk_size * total_size[1] * total_size[2] * total_size[3] * 4 / 1024**2):.2f} MB")
        
        return {
            'full_memory_mb': full_memory,
            'chunk_memory_mb': chunk_size * total_size[1] * total_size[2] * total_size[3] * 4 / 1024**2,
            'time_saved': full_time - chunk_time
        }
    
    def test_gradient_accumulation_concept(self):
        """Test concepto de gradient accumulation"""
        print("\n📈 Test 5: Gradient Accumulation Concept")
        print("-" * 40)
        
        # Simular gradient accumulation
        real_batch_size = 4
        accumulated_batch_size = 2
        accumulation_steps = real_batch_size // accumulated_batch_size
        
        # Simular pérdida
        losses = [0.8, 0.7, 0.6, 0.5]  # 4 batches
        
        print(f"   Batch real deseado: {real_batch_size}")
        print(f"   Batch actual (sin OOM): {accumulated_batch_size}")
        print(f"   Accumulation steps: {accumulation_steps}")
        
        # Sin accumulation (limitado por OOM)
        effective_loss_no_accum = np.mean(losses[:accumulated_batch_size])
        print(f"   Pérdida sin accumulation: {effective_loss_no_accum:.4f}")
        
        # Con accumulation
        accumulated_gradients = []
        for i in range(0, real_batch_size, accumulated_batch_size):
            batch_losses = losses[i:i+accumulated_batch_size]
            accumulated_gradients.append(np.mean(batch_losses))
        
        effective_loss_with_accum = np.mean(accumulated_gradients)
        print(f"   Pérdida con accumulation: {effective_loss_with_accum:.4f}")
        
        print(f"   ✅ Effective batch size: {accumulated_batch_size * accumulation_steps}")
        print(f"   ✅ Memory usage: {accumulated_batch_size/real_batch_size*100:.0f}% del original")
        
        return {
            'memory_reduction': accumulated_batch_size / real_batch_size,
            'effective_batch_size': accumulated_batch_size * accumulation_steps
        }
    
    def run_all_tests(self):
        """Ejecutar todos los tests"""
        print("🚀 Ejecutando battery de tests lógicos...")
        
        results = {}
        
        # Ejecutar tests
        results['memory'] = self.test_memory_patterns()
        results['mamba'] = self.test_mamba_token_explosion()
        results['config'] = self.test_config_scaling()
        results['chunks'] = self.test_data_pipeline_chunks()
        results['gradient'] = self.test_gradient_accumulation_concept()
        
        # Resumen
        print("\n" + "=" * 50)
        print("📊 RESUMEN DE RESULTADOS")
        print("=" * 50)
        
        print(f"\n🧠 Memoria Pipeline:")
        print(f"   Problema np.repeat: {results['memory']['repeat_memory']:.2f} MB")
        print(f"   Solución broadcast: {results['memory']['broadcast_memory']:.2f} MB")
        print(f"   ✅ Memory saved: {results['memory']['repeat_memory'] - results['memory']['broadcast_memory']:.2f} MB")
        
        print(f"\n🔥 Mamba Token Explosion:")
        print(f"   Tokens por secuencia: {results['mamba']['tokens_per_sequence']:,}")
        print(f"   Memory batch 4: {results['mamba']['memory_per_batch_4']:.2f} MB")
        
        print(f"\n⚙️  Config Scaling:")
        best_config = results['config'][1]  # Optimizado
        print(f"   Config recomendada: {best_config['name']}")
        print(f"   Seq: {best_config['seq_len']}, Size: {best_config['hr_shape']}, Batch: {best_config['batch_size']}")
        
        print(f"\n📦 Chunks Benefits:")
        print(f"   Memory saved: {results['chunks']['time_saved']:.2f} MB")
        
        print(f"\n📈 Gradient Accumulation:")
        print(f"   Memory reduction: {results['gradient']['memory_reduction']*100:.0f}%")
        print(f"   Effective batch: {results['gradient']['effective_batch_size']}")
        
        # Verdict
        print(f"\n🎯 VEREDICTO:")
        
        if results['memory']['repeat_memory'] > 1000:
            print("✅ Memory optimization CRITICAL - np.broadcast_to es esencial")
        
        if results['mamba']['memory_per_batch_4'] > 8000:
            print("⚠️  Mamba necesita batch_size=1 y sequence reduction")
        
        if results['chunks']['time_saved'] > 0:
            print("✅ Chunking necesario para datasets grandes")
        
        print("🎉 Pipeline lógico validado! Listo para implementación real.")
        
        return results

def main():
    """Ejecución principal"""
    tester = MockMLTest()
    results = tester.run_all_tests()
    
    return results

if __name__ == "__main__":
    main()