### Caratteristiche principali:

1. **Architettura della rete**:
    - Neuroni completamente interconnessi
    - Pesi simmetrici (w_ij = w_ji)
    - Nessuna auto-connessione (w_ii = 0)
2. **Addestramento**:
    - Implementa la regola di Hebb
    - Normalizzazione dei pesi
    - Supporta pattern binari (-1, 1)
3. **Modalità di aggiornamento**:
    - **Asincrona**: Aggiornamento casuale dei neuroni
    - **Sincrona**: Aggiornamento simultaneo di tutti i neuroni
4. **Funzionalità di analisi**:
    - Calcolo dell'energia della rete
    - Tracciamento dell'evoluzione degli stati
    - Visualizzazione della matrice dei pesi
5. **Esempio dimostrativo**:
    - Ricostruzione di pattern binari distorti
    - Confronto tra pattern originale, rumoroso e ricostruito
    - Visualizzazione dell'andamento energetico

### Istruzioni per l'uso:

1. Creare un'istanza della rete con `hopfield = HopfieldNetwork(n_neurons)`
2. Addestrare con i pattern: `hopfield.train(patterns_list)`
3. Ricostruire un pattern: `recovered, history, energy = hopfield.predict(noisy_pattern)`

### Dipendenze:

- NumPy
- Matplotlib
- tqdm (per le barre di avanzamento)

Questo programma implementa tutte le caratteristiche fondamentali delle reti di Hopfield come descritto nei risultati di ricerca, inclusi:

- Memorizzazione di pattern
- Recupero associativo da input distorti
- Dinamica asincrona/sincrona
- Funzione energetica
- Visualizzazione dei risultati

<div style="text-align: center">⁂</div>

[^1]: https://sarunthapa.hashnode.dev/hopfield-neural-network

[^2]: https://github.com/zftan0709/Hopfield-Network

[^3]: https://www.i-programmer.info/programming/102-artificial-intelligence/5097-the-hopfield-network-a-simple-python-example.html

[^4]: https://neuronaldynamics-exercises.readthedocs.io/en/latest/exercises/hopfield-network.html

[^5]: https://github.com/takyamamoto/Hopfield-Network

[^6]: https://www.i-programmer.info/programming/102-artificial-intelligence/5097-the-hopfield-network-a-simple-python-example.html?start=1

[^7]: https://github.com/mburakbozbey/hopfield-python

[^8]: https://github.com/zftan0709/Hopfield-Network/blob/master/README.md

[^9]: https://www.i-programmer.info/programming/102-artificial-intelligence/5097-the-hopfield-network-a-simple-python-example.html?start=2

[^10]: https://datascience.stackexchange.com/questions/94665/hopfield-network-python-implementation-network-doesnt-converge-to-one-of-the-l

