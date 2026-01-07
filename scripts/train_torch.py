import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from torch_engine.model_mamba import DownsrUNetMamba

def train():
    print("🚀 Iniciando Entrenamiento Mamba (PyTorch)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Dispositivo: {device}")
    
    model = DownsrUNetMamba().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print("⚠️ Usando datos dummy para prueba de integridad...")
    dummy_dyn = torch.randn(8, 9, 64, 64).to(device)
    dummy_static = torch.randn(8, 4, 64, 64).to(device)
    dummy_target = torch.randn(8, 1, 64, 64).to(device)
    
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        output = model(dummy_dyn, dummy_static)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        print(f"Epoca {epoch+1} | Loss: {loss.item():.6f}")
    print("✅ Mamba funcionando OK.")

if __name__ == "__main__":
    train()
