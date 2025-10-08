#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from funzioni_attivazione import sigmoid, sigmoid_derivative

# ============================================================================
# MULTI-LAYER PERCEPTRON (MLP) - IMPLEMENTAZIONE SENZA CLASSI
# ============================================================================

def inizializza_mlp(input_size, hidden_size, output_size):
    """
    Inizializza i pesi e bias per un MLP con un solo strato nascosto
    
    Args:
        input_size: numero di neuroni di input
        hidden_size: numero di neuroni nello strato nascosto
        output_size: numero di neuroni di output
    
    Returns:
        weights_input_hidden: matrice pesi input -> hidden
        bias_hidden: bias dello strato nascosto
        weights_hidden_output: matrice pesi hidden -> output
        bias_output: bias dello strato di output
    """
    # Inizializzazione Xavier/Glorot per migliore convergenza
    weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size)) * np.sqrt(2.0 / input_size)
    bias_hidden = np.zeros(hidden_size)
    
    weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
    bias_output = np.zeros(output_size)
    
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

def forward_pass_mlp(inputs, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    """
    Esegue il forward pass attraverso la rete MLP
    
    Il forward pass calcola l'output della rete propagando l'input
    attraverso tutti gli strati.
    
    Passaggi:
    1. Input -> Strato nascosto: z1 = input * W1 + b1, a1 = sigmoid(z1)
    2. Strato nascosto -> Output: z2 = a1 * W2 + b2, a2 = sigmoid(z2)
    
    Args:
        inputs: vettore di input
        weights_input_hidden: pesi input -> hidden
        bias_hidden: bias hidden
        weights_hidden_output: pesi hidden -> output
        bias_output: bias output
    
    Returns:
        output: output finale della rete
        hidden_output: output dello strato nascosto (per backprop)
        hidden_input: input dello strato nascosto prima dell'attivazione
        output_input: input dello strato di output prima dell'attivazione
    """
    # Strato nascosto
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    
    # Strato di output
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = sigmoid(output_input)
    
    return output, hidden_output, hidden_input, output_input

def backward_pass_mlp(inputs, target, output, hidden_output, hidden_input, output_input,
                     weights_input_hidden, weights_hidden_output):
    """
    Esegue il backward pass (backpropagation) per calcolare i gradienti
    
    La backpropagation calcola i gradienti dell'errore rispetto ai pesi
    propagando l'errore all'indietro attraverso la rete.
    
    Formule chiave:
    1. Errore output: δ2 = (target - output) * sigmoid'(output_input)
    2. Errore hidden: δ1 = δ2 * W2 * sigmoid'(hidden_input)
    3. Gradienti: ∇W = input * δ, ∇b = δ
    
    Args:
        inputs: input originale
        target: output desiderato
        output: output attuale della rete
        hidden_output: output dello strato nascosto
        hidden_input: input dello strato nascosto (pre-attivazione)
        output_input: input dello strato di output (pre-attivazione)
        weights_input_hidden: pesi input -> hidden
        weights_hidden_output: pesi hidden -> output
    
    Returns:
        grad_weights_input_hidden: gradiente pesi input -> hidden
        grad_bias_hidden: gradiente bias hidden
        grad_weights_hidden_output: gradiente pesi hidden -> output
        grad_bias_output: gradiente bias output
    """
    # Calcola l'errore per lo strato di output
    output_error = target - output
    output_delta = output_error * sigmoid_derivative(output_input)
    
    # Calcola i gradienti per i pesi e bias dello strato di output
    grad_weights_hidden_output = np.outer(hidden_output, output_delta)
    grad_bias_output = output_delta
    
    # Propaga l'errore allo strato nascosto
    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_input)
    
    # Calcola i gradienti per i pesi e bias dello strato nascosto
    grad_weights_input_hidden = np.outer(inputs, hidden_delta)
    grad_bias_hidden = hidden_delta
    
    return grad_weights_input_hidden, grad_bias_hidden, grad_weights_hidden_output, grad_bias_output

def train_mlp(X, y, hidden_size=4, learning_rate=1.0, max_epochs=10000, tolerance=1e-6):
    """
    Addestra il MLP usando backpropagation
    
    Args:
        X: matrice degli input di training
        y: vettore dei target
        hidden_size: numero di neuroni nello strato nascosto
        learning_rate: tasso di apprendimento
        max_epochs: numero massimo di epoche
        tolerance: tolleranza per la convergenza
    
    Returns:
        weights_input_hidden: pesi finali input -> hidden
        bias_hidden: bias finali hidden
        weights_hidden_output: pesi finali hidden -> output
        bias_output: bias finali output
        errors_per_epoch: lista degli errori per epoca
    """
    n_samples, n_features = X.shape
    input_size = n_features
    output_size = 1
    
    # Inizializza la rete
    weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = \
        inizializza_mlp(input_size, hidden_size, output_size)
    
    errors_per_epoch = []
    
    print(f"Inizializzazione MLP:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Output size: {output_size}")
    print(f"  Learning rate: {learning_rate}")
    print("-" * 50)
    
    for epoch in range(max_epochs):
        total_error = 0
        
        # Addestramento su tutti gli esempi
        for i in range(n_samples):
            # Forward pass
            output, hidden_output, hidden_input, output_input = \
                forward_pass_mlp(X[i], weights_input_hidden, bias_hidden, 
                               weights_hidden_output, bias_output)
            
            # Calcola l'errore
            error = 0.5 * (y[i] - output[0]) ** 2
            total_error += error
            
            # Backward pass
            grad_w_ih, grad_b_h, grad_w_ho, grad_b_o = \
                backward_pass_mlp(X[i], y[i], output, hidden_output, hidden_input, output_input,
                                weights_input_hidden, weights_hidden_output)
            
            # Aggiorna i pesi e bias
            weights_input_hidden += learning_rate * grad_w_ih
            bias_hidden += learning_rate * grad_b_h
            weights_hidden_output += learning_rate * grad_w_ho
            bias_output += learning_rate * grad_b_o
        
        errors_per_epoch.append(total_error)
        
        # Stampa progresso ogni 1000 epoche o se converge
        if epoch % 1000 == 0 or total_error < tolerance:
            print(f"Epoca {epoch+1}: Errore totale = {total_error:.6f}")
        
        # Controlla convergenza
        if total_error < tolerance:
            print(f"Convergenza raggiunta in {epoch+1} epoche!")
            break
    
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, errors_per_epoch

def test_mlp(X, y, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    """
    Testa il MLP su un set di dati
    
    Args:
        X: matrice degli input di test
        y: vettore dei target di test
        weights_input_hidden: pesi addestrati input -> hidden
        bias_hidden: bias addestrati hidden
        weights_hidden_output: pesi addestrati hidden -> output
        bias_output: bias addestrati output
    
    Returns:
        accuracy: accuratezza del MLP
        predictions: predizioni del MLP
    """
    predictions = []
    correct = 0
    
    print("=== RISULTATI DEL TEST MLP ===")
    for i in range(len(X)):
        output, _, _, _ = forward_pass_mlp(X[i], weights_input_hidden, bias_hidden,
                                         weights_hidden_output, bias_output)
        
        # Converte output continuo in predizione binaria
        prediction = 1 if output[0] >= 0.5 else 0
        predictions.append(prediction)
        
        is_correct = (prediction == y[i])
        if is_correct:
            correct += 1
            
        print(f"Input: {X[i]} -> Output: {output[0]:.4f}, Predizione: {prediction}, "
              f"Target: {y[i]}, Corretto: {is_correct}")
    
    accuracy = correct / len(X)
    print(f"\nAccuratezza: {accuracy:.2%} ({correct}/{len(X)})")
    
    return accuracy, predictions

def crea_dataset_xor():
    """Crea il dataset per la funzione XOR"""
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)
    return X, y

def visualizza_superficie_decisione_mlp(weights_input_hidden, bias_hidden, 
                                       weights_hidden_output, bias_output):
    """
    Visualizza la superficie di decisione del MLP
    """
    # Crea una griglia di punti
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Calcola l'output per ogni punto della griglia
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    
    for point in grid_points:
        output, _, _, _ = forward_pass_mlp(point, weights_input_hidden, bias_hidden,
                                         weights_hidden_output, bias_output)
        Z.append(output[0])
    
    Z = np.array(Z).reshape(xx.shape)
    
    # Crea il plot
    plt.figure(figsize=(10, 8))
    
    # Superficie di decisione
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Output MLP')
    
    # Linea di decisione (output = 0.5)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    # Punti del dataset XOR
    X_xor, y_xor = crea_dataset_xor()
    colors = ['red', 'blue']
    for i in range(len(X_xor)):
        plt.scatter(X_xor[i, 0], X_xor[i, 1], c=colors[int(y_xor[i])], s=200, 
                   edgecolors='black', linewidth=2)
        plt.annotate(f'({X_xor[i, 0]:.0f},{X_xor[i, 1]:.0f})→{y_xor[i]:.0f}', 
                    (X_xor[i, 0], X_xor[i, 1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12, fontweight='bold')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('MLP - Superficie di Decisione per XOR')
    plt.grid(True, alpha=0.3)
    
    # Legenda
    plt.scatter([], [], c='red', s=100, label='Classe 0', edgecolors='black')
    plt.scatter([], [], c='blue', s=100, label='Classe 1', edgecolors='black')
    plt.legend()
    
    plt.savefig('/home/ubuntu/mlp_superficie_decisione.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Funzione principale che dimostra il MLP su XOR
    """
    print("=" * 60)
    print("MULTI-LAYER PERCEPTRON - SOLUZIONE DEL PROBLEMA XOR")
    print("=" * 60)
    
    # Crea il dataset XOR
    X_xor, y_xor = crea_dataset_xor()
    
    print("Dataset XOR:")
    for i in range(len(X_xor)):
        print(f"  {X_xor[i]} -> {y_xor[i]}")
    
    # Addestra il MLP
    print(f"\nAddestramento MLP...")
    weights_ih, bias_h, weights_ho, bias_o, errors = \
        train_mlp(X_xor, y_xor, hidden_size=4, learning_rate=10.0, max_epochs=10000)
    
    # Testa il MLP
    accuracy, predictions = test_mlp(X_xor, y_xor, weights_ih, bias_h, weights_ho, bias_o)
    
    # Visualizza la superficie di decisione
    visualizza_superficie_decisione_mlp(weights_ih, bias_h, weights_ho, bias_o)
    
    # Grafico dell'errore durante l'addestramento
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'b-', linewidth=2)
    plt.title('Errore durante addestramento MLP - XOR')
    plt.xlabel('Epoca')
    plt.ylabel('Errore totale')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/ubuntu/mlp_training_error.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n" + "=" * 60)
    print("RISULTATO:")
    if accuracy >= 0.99:
        print("✓ Il MLP ha risolto con successo il problema XOR!")
        print("  Le reti multi-strato possono apprendere funzioni non linearmente separabili.")
    else:
        print("? Risultato inaspettato - verificare i parametri di addestramento")
    print("=" * 60)

if __name__ == "__main__":
    main()

