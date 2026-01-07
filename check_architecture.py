import os
import sys
import tensorflow as tf

# Aseguramos que Python vea la carpeta src
sys.path.append(os.path.join(os.getcwd(), 'src'))

print(f"🔍 Iniciando Diagnóstico de Arquitectura DOWNSR (TensorFlow)...")

try:
    from src.models.downsr_unet import DownsrUNet
    print("✅ Clase 'DownsrUNet' importada correctamente.")
except ImportError as e:
    print(f"❌ ERROR CRÍTICO DE IMPORTACIÓN: {e}")
    sys.exit(1)

# Configuraciones Dummy (Tal cual espera tu Config)
# Dyn: (Time, H_lr, W_lr, Channels) -> (3, 5, 9, 9)
SHAPE_DYN = (3, 5, 9, 9)       
# Static: (Time, H_hr, W_hr, Channels) -> (3, 251, 251, 4)
SHAPE_ST = (3, 251, 251, 4)   

def test_build(strategy_name):
    print(f"\n🛠️  Construyendo variante: {strategy_name.upper()}...")
    try:
        # CORRECCIÓN AQUÍ: Usamos los nombres que definió el Agente (dyn/st)
        model_builder = DownsrUNet(
            input_shape_dyn=SHAPE_DYN,
            input_shape_st=SHAPE_ST,
            bottleneck_type=strategy_name
        )
        model = model_builder.build()
        
        # Verificamos si compila
        print(f"   ✅ Modelo {strategy_name} construido. Params: {model.count_params():,}")
        return True
    except Exception as e:
        # Si falla, imprimimos el error completo para debuggear
        import traceback
        traceback.print_exc()
        return False

# Ejecutar pruebas
success_mamba = test_build('mamba') 
success_lstm = test_build('lstm')
success_base = test_build('unet')

if success_mamba and success_lstm:
    print("\n🎉 ÉXITO TOTAL: La arquitectura modular es estable.")
else:
    print("\n⚠️ ALERTA: Fallo en la construcción interna. Revisa el Traceback.")