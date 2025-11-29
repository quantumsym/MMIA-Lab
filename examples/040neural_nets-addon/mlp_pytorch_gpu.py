# pytorch_gpu_mlp.py
import torch
import torch.nn as nn
import time

# 1. Setup del dispositivo
# Seleziona la GPU se disponibile, altrimenti la CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Creazione dei dati
# Creiamo i dati direttamente sul dispositivo target per efficienza
x_train = torch.linspace(-2, 2, 2000, device=device).unsqueeze(1)
y_train = x_train3 - 2 * x_train2 + x_train

# 3. Definizione del Modello (OOP)
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# 4. Inizializzazione e Spostamento su GPU
model = SimpleMLP()
# Spostiamo esplicitamente i parametri del modello sulla GPU
model.to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. Ciclo di Addestramento
print("\nStarting PyTorch training...")
start_time = time.time()

for epoch in range(2001):
    # Il forward pass viene eseguito sulla GPU perché sia il modello che i dati sono lì
    predictions = model(x_train)
    loss = loss_function(predictions, y_train)

    # Backward pass e ottimizzazione
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

end_time = time.time()
print(f"PyTorch training finished in {end_time - start_time:.2f} seconds.")
```

