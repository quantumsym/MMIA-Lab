# jax_gpu_mlp.py
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import time

# 1. Setup del dispositivo
# JAX rileva e utilizza automaticamente la GPU se disponibile
print(f"JAX is using: {jax.default_backend().upper()}")

# 2. Creazione dei dati
# Usiamo jax.device_put per assicurarci che i dati siano sul dispositivo corretto
device = jax.devices()[0]
x_train = jax.device_put(jnp.linspace(-2, 2, 2000).reshape(-1, 1), device)
y_train = x_train3 - 2 * x_train2 + x_train

# 3. Definizione del Modello (Funzionale)
class SimpleMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

# 4. Funzione di Addestramento Pura e Compilata JIT
# Questa funzione viene compilata da XLA per l'esecuzione su GPU
@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        predictions = model.apply({'params': params}, batch_x)
        loss = jnp.mean((predictions - batch_y)  2)
        return loss

    # Calcola i gradienti
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # Aggiorna lo stato (parametri e ottimizzatore)
    state = state.apply_gradients(grads=grads)
    return state, loss

# 5. Inizializzazione dello Stato
key = jax.random.PRNGKey(0)
model = SimpleMLP()
params = model.init(key, x_train)['params']

optimizer = optax.adam(1e-3)
# Lo 'stato' contiene i parametri e lo stato dell'ottimizzatore
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

# 6. Ciclo di Addestramento
print("\nStarting JAX training...")
start_time = time.time()

for epoch in range(2001):
    # La funzione train_step viene eseguita interamente su GPU grazie a @jax.jit
    state, loss = train_step(state, x_train, y_train)
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

end_time = time.time()
print(f"JAX training finished in {end_time - start_time:.2f} seconds.")
```

