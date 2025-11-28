## Documentazione degli Script Procedurali del Modello di Hopfield

Questo documento fornisce una panoramica di ciascuno dei quattro script procedurali, spiegandone lo scopo, i risultati attesi e le principali funzioni NumPy utilizzate.

---

### 1. `monte_carlo_hopfield_procedural.py`

#### Introduzione

Questo script implementa una **rete di Hopfield stocastica** utilizzando una simulazione **Monte Carlo**. A differenza di un modello puramente deterministico, questo approccio introduce un elemento di casualità controllato dal parametro "temperatura" (T). Le transizioni di stato dei neuroni non avvengono in modo deterministico, ma seguono la **statistica di Boltzmann**:

- Le transizioni che **diminuiscono l'energia** del sistema vengono **sempre accettate**.
- Le transizioni che **aumentano l'energia** possono essere **accettate con una probabilità** `exp(-ΔE / T)`.

Questo meccanismo probabilistico è fondamentale perché permette alla rete di "sfuggire" a minimi locali di energia (stati sub-ottimali) e di esplorare più a fondo il paesaggio energetico alla ricerca di una configurazione globalmente migliore. Lo script è la base per le tecniche di ottimizzazione più avanzate come il Simulated Annealing e il Parallel Tempering.

#### Funzioni NumPy Utilizzate

| Funzione | Significato e Utilizzo |
| :--- | :--- |
| `np.zeros` | Crea un array riempito di zeri. Usato per inizializzare la matrice dei pesi (`weights`) e il vettore delle soglie (`thresholds`). |
| `np.random.choice` | Genera un campione casuale da un dato array. Usato per inizializzare lo stato dei neuroni in modo casuale a -1 o +1. |
| `np.outer` | Calcola il prodotto esterno di due vettori. È il cuore della **regola di apprendimento di Hebb** per costruire la matrice dei pesi a partire dai pattern da memorizzare. |
| `np.fill_diagonal` | Riempie la diagonale principale di un array con un valore specificato. Usato per impostare a zero le connessioni di un neurone con se stesso, una condizione standard per le reti di Hopfield. |
| `np.dot` | Calcola il prodotto scalare tra due array. Usato per calcolare l'energia del sistema e il "campo locale" che agisce su un neurone. |
| `np.random.randint` | Restituisce interi casuali da un intervallo. Usato per selezionare casualmente un neurone da "flippare" durante ogni passo della simulazione Monte Carlo. |
| `np.exp` | Calcola l'esponenziale. Usato per calcolare la **probabilità di accettazione di Metropolis** per le transizioni che aumentano l'energia. |
| `np.random.rand` | Crea un campione casuale da una distribuzione uniforme su [0, 1). Usato per confrontare con la probabilità di accettazione e decidere se effettuare una transizione sfavorevole. |
| `np.array` | Crea un array NumPy. Usato per convertire le liste (es. la cronologia delle energie) in array per una manipolazione più efficiente. |

---

### 2. `simulated_annealing_hopfield_procedural.py`

#### Introduzione

Questo script implementa due tecniche di ottimizzazione stocastica avanzate, costruite sulla base della simulazione Monte Carlo:

1.  **Simulated Annealing (SA)**: Questa tecnica migliora la ricerca del minimo globale introducendo un **programma di raffreddamento dinamico**. La simulazione inizia ad una temperatura elevata, permettendo al sistema di superare facilmente le barriere energetiche e di esplorare l'intero spazio degli stati. Successivamente, la temperatura viene abbassata gradualmente. Questo processo, analogo al raffreddamento lento di un metallo (annealing), guida il sistema ad assestarsi in stati di energia sempre più bassi, aumentando significativamente la probabilità di convergere verso il minimo globale anziché rimanere intrappolato in un minimo locale.

2.  **Parallel Tempering (PT)**: Questa è una tecnica ancora più potente che esegue **simultaneamente più simulazioni Monte Carlo** (chiamate "repliche") a diverse temperature fisse, da molto bassa a molto alta. Periodicamente, viene tentato uno **scambio di configurazioni** tra repliche a temperature adiacenti. Questo meccanismo permette alle soluzioni "bloccate" a bassa temperatura di "scaldarsi" per sfuggire ai minimi locali, e alle buone configurazioni trovate ad alte temperature di "raffreddarsi" per essere rifinite con precisione. È particolarmente efficace per paesaggi energetici complessi e "frastagliati".

#### Funzioni NumPy Utilizzate

| Funzione | Significato e Utilizzo |
| :--- | :--- |
| `np.geomspace` | Restituisce numeri spaziati uniformemente su una **scala logaritmica**. È ideale per creare il "ladder" di temperature per il Parallel Tempering, garantendo che ci siano più repliche a basse temperature dove l'esplorazione è più critica. |
| `np.log` | Calcola il logaritmo naturale. Usato per implementare lo schema di raffreddamento logaritmico nel Simulated Annealing, uno dei possibili modi per diminuire la temperatura. |
| `np.mean` | Calcola la media aritmetica. Usato per calcolare il tasso di scambio medio tra le repliche nel Parallel Tempering, un'importante metrica di performance dell'algoritmo. |
| `np.isfinite` | Testa se gli elementi di un array sono finiti (non `inf`, `-inf` o `nan`). Utile per gestire in modo robusto eventuali divisioni per zero nel calcolo dei tassi di scambio. |
| `np.sum` | Somma gli elementi di un array. Usato per calcolare il tasso di scambio complessivo in tutto il sistema Parallel Tempering. |
| *Altre* | Include anche le funzioni di base di `monte_carlo_hopfield_procedural.py` (`np.exp`, `np.random.rand`, etc.) per eseguire i passi Monte Carlo all'interno di ogni replica. |

---

### 3. `mean_field_hopfield_procedural.py`

#### Introduzione

Questo script adotta un approccio completamente diverso per analizzare le reti di Hopfield, basato sulla **Teoria di Campo Medio (Mean Field Theory, MFT)**, un potente strumento della fisica statistica. Invece di simulare lo stato binario (`-1` o `+1`) di ogni singolo neurone, la MFT descrive il sistema attraverso le **"magnetizzazioni" medie** (`m_i`), che sono variabili **continue** tra -1 e +1 e rappresentano lo stato medio di ciascun neurone.

Questo approccio trasforma un complesso problema stocastico e discreto in un sistema di **equazioni continue e deterministiche**. La soluzione di queste equazioni (`m_i = tanh(h_i / T)`) fornisce una descrizione approssimata ma estremamente utile delle proprietà macroscopiche del sistema, come:

-   Gli **stati di equilibrio** (attrattori o fixed points).
-   L'**energia libera** del sistema, un concetto termodinamico che bilancia energia ed entropia.
-   Le **transizioni di fase**, ad esempio il passaggio da uno stato disordinato a uno stato in cui la rete "ricorda" un pattern.

Il vantaggio principale è il **costo computazionale molto inferiore** rispetto alle simulazioni stocastiche, che permette un'analisi rapida e analitica del comportamento della rete.

#### Funzioni NumPy Utilizzate

| Funzione | Significato e Utilizzo |
| :--- | :--- |
| `np.random.uniform` | Estrae campioni da una distribuzione uniforme. Usato per inizializzare le magnetizzazioni medie con piccoli valori casuali per rompere la simmetria iniziale. |
| `np.tanh` | Calcola la **tangente iperbolica**. È la funzione centrale della MFT, che lega il campo medio (`h`) alla magnetizzazione di equilibrio (`m`): `m = tanh(h/T)`. |
| `np.sign` | Restituisce il segno di un array, elemento per elemento. Rappresenta il comportamento deterministico della rete nel limite a temperatura zero (`T → 0`), dove `tanh(x) → sign(x)`. |
| `np.max(np.abs(...))` | Trova il massimo cambiamento assoluto tra le magnetizzazioni di due iterazioni successive. È il **criterio di convergenza** per l'algoritmo iterativo che risolve le equazioni di campo medio. |
| `np.clip` | Limita i valori in un array a un intervallo specificato. Usato per evitare errori numerici (es. `log(0)`) nel calcolo dell'entropia quando le magnetizzazioni si avvicinano ai valori estremi +1 o -1. |
| `np.linalg.norm` | Calcola la norma di un vettore (es. la norma L2). Usato per misurare la "forza" complessiva del vettore delle magnetizzazioni, utile per identificare le transizioni di fase. |
| `np.diff` | Calcola la differenza tra elementi adiacenti in un array. Usato per rilevare grandi cambiamenti nella norma della magnetizzazione al variare della temperatura, che segnalano una transizione di fase. |
| `np.argmax` | Restituisce l'indice del valore massimo. Usato per trovare la **temperatura critica** approssimativa dove avviene la transizione di fase. |

---

### 4. `hopfield_tsp_optimization_procedural.py`

#### Introduzione

Questo script applica le reti di Hopfield per risolvere un classico problema di ottimizzazione combinatoria: il **Problema del Commesso Viaggiatore (Traveling Salesman Problem, TSP)**. L'obiettivo del TSP è trovare il percorso più breve che visiti un dato insieme di città una sola volta e ritorni alla città di partenza.

L'idea chiave è **mappare il problema del TSP su una funzione di energia** della rete di Hopfield. Questo viene fatto nel seguente modo:

1.  **Rappresentazione**: Si usa una rete con `N x N` neuroni, dove `N` è il numero di città. L'attivazione del neurone `(i, j)` significa che la città `i` viene visitata nella posizione `j` del tour.
2.  **Funzione di Energia**: La matrice dei pesi e le soglie sono costruite in modo tale che la funzione di energia della rete codifichi sia i **vincoli** del TSP (ogni città visitata una sola volta, ogni posizione occupata una sola volta) sia l'**obiettivo** (minimizzare la distanza totale del percorso).

Una volta definita la rete, gli stati di energia minima corrispondono a tour validi e brevi. Lo script utilizza le tecniche avanzate di **Simulated Annealing** e **Parallel Tempering** per trovare soluzioni di alta qualità e le confronta con un approccio Monte Carlo standard e, per piccoli problemi, con la soluzione ottimale calcolata tramite forza bruta.

#### Funzioni Scientifiche Utilizzate

| Funzione | Libreria | Significato e Utilizzo |
| :--- | :--- | :--- |
| `pdist` | `scipy.spatial.distance` | Calcola la distanza tra tutte le coppie di punti in un set di dati. Usato per calcolare le distanze tra tutte le città in modo efficiente. |
| `squareform` | `scipy.spatial.distance` | Converte un vettore di distanze in forma "condensata" (prodotto da `pdist`) in una matrice di distanze quadrata e simmetrica, più facile da usare. |
| `permutations` | `itertools` | Genera tutte le possibili permutazioni di una sequenza. Usato per l'approccio a **forza bruta** per trovare il tour ottimale, che è fattibile solo per un numero molto piccolo di città a causa della sua complessità fattoriale `O(N!)`. |
| `np.reshape` | `numpy` | Cambia la forma di un array senza cambiarne i dati. Fondamentale per convertire lo stato lineare della rete (un vettore di `N*N` elementi) in una matrice `N×N` per l'analisi e la decodifica del tour. |
| `np.argsort` | `numpy` | Restituisce gli indici che ordinerebbero un array. Usato per trovare le città più "attive" in ordine decrescente quando si deve costruire un tour approssimato da uno stato della rete che non è perfettamente valido. |

