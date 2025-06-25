# Programma Python per la simulazione del modello Sherrington-Kirkpatrick con metodi Monte Carlo

Il modello Sherrington-Kirkpatrick (SK) è un modello mean-field per i vetri di spin, introdotto nel 1975 da David Sherrington e Scott Kirkpatrick[^1]. Questo modello rappresenta un sistema di spin di Ising con interazioni casuali a lungo raggio, caratterizzato da un comportamento magnetico disordinato e dalla presenza di frustrazioni geometriche[^2]. Ecco un'implementazione completa in Python che utilizza metodi Monte Carlo avanzati per simulare il modello SK.

## Formulazione del modello SK

L'Hamiltoniana del modello Sherrington-Kirkpatrick è definita come[^3][^4]:

\$ H_N(\sigma) = -\frac{1}{\sqrt{N}} \sum_{i,j=1}^{N} g_{ij}\sigma_i\sigma_j \$

dove $\sigma_i = \pm 1$ sono gli spin di Ising, $g_{ij}$ sono variabili casuali gaussiane indipendenti con media zero e varianza unitaria, e la normalizzazione $1/\sqrt{N}$ mantiene l'energia per spin di ordine 1[^5][^4].

## Caratteristiche del programma

### **Algoritmi Monte Carlo implementati**

Il programma include tre algoritmi Monte Carlo fondamentali per la simulazione di vetri di spin[^6][^7]:

1. **Algoritmo di Metropolis**: Implementa il criterio di accettazione standard per esplorare lo spazio delle configurazioni con probabilità di accettazione P = min(1, exp(-βΔE))[^8]
2. **Heat Bath**: Algoritmo più efficiente a basse temperature che campiona direttamente dalla distribuzione di Boltzmann locale[^8]
3. **Simulated Annealing**: Tecnica di ottimizzazione che raffredda gradualmente il sistema per trovare configurazioni a bassa energia[^9][^10]
4. **Parallel Tempering**: Metodo avanzato che simula multiple copie del sistema a temperature diverse, permettendo scambi periodici per migliorare l'esplorazione[^11]

### **Osservabili fisiche calcolate**

Il programma calcola automaticamente le principali grandezze termodinamiche caratteristiche dei vetri di spin[^2][^12]:

- **Energia media**: Valore dell'Hamiltoniana nel regime di equilibrio
- **Magnetizzazione**: Momento magnetico medio del sistema
- **Calore specifico**: Fluttuazioni dell'energia, indicatore delle transizioni di fase
- **Suscettibilità magnetica**: Risposta del sistema a campi esterni
- **Parametro di overlap**: Misura delle correlazioni tra configurazioni diverse, cruciale per caratterizzare la fase vetrosa[^13][^14]


### **Analisi delle transizioni di fase**

Il modello SK presenta una transizione di fase da una fase paramagnetica ad alta temperatura a una fase vetrosa a bassa temperatura[^2][^15]. La temperatura critica teorica è Tc = 1 (in unità adimensionali), e il programma permette di studiare questa transizione attraverso il comportamento degli osservabili.

## Vantaggi dell'implementazione

L'approccio Monte Carlo per il modello SK offre diversi vantaggi rispetto ai metodi analitici[^16][^17]:

- **Flessibilità**: Possibilità di studiare diverse realizzazioni del disordine e diversi parametri
- **Controllo completo**: Accesso a tutte le configurazioni microscopiche durante l'evoluzione
- **Metodi avanzati**: Implementazione di tecniche sofisticate come parallel tempering per superare le barriere energetiche
- **Visualizzazione**: Analisi in tempo reale dell'evoluzione del sistema e delle distribuzioni di probabilità

Questo programma fornisce una base solida per lo studio computazionale dei vetri di spin SK, permettendo di esplorare sia il regime di alta temperatura (fase paramagnetica) che quello di bassa temperatura (fase vetrosa), caratteristico di questi sistemi frustrati[^2][^4].

<div style="text-align: center">⁂</div>

[^1]: https://link.aps.org/doi/10.1103/PhysRevLett.35.1792

[^2]: https://en.wikipedia.org/wiki/Spin_glass

[^3]: https://annals.math.princeton.edu/wp-content/uploads/annals-v163-n1-p04.pdf

[^4]: http://arxiv.org/pdf/1211.1094.pdf

[^5]: https://www.intlpress.com/site/pub/files/_fulltext/journals/cdm/2014/2014/0001/CDM-2014-2014-0001-a004.pdf

[^6]: https://arxiv.org/abs/2210.11288

[^7]: https://arxiv.org/abs/2006.08378

[^8]: https://magnetism.eu/esm/2009/slides/neda-tutorial-1.pdf

[^9]: https://arxiv.org/html/2309.11822v3

[^10]: https://www.cs.amherst.edu/~ccmcgeoch/cs34/papers/parkkimsdarticle-5.pdf

[^11]: https://bpb-us-e1.wpmucdn.com/sites.ucsc.edu/dist/7/1905/files/2025/03/erice.pdf

[^12]: https://core.ac.uk/download/pdf/73346567.pdf

[^13]: https://arxiv.org/pdf/math/0604082.pdf

[^14]: https://projecteuclid.org/journals/communications-in-mathematical-physics/volume-90/issue-3/The-order-parameter-in-a-spin-glass/cmp/1103940344.pdf

[^15]: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.034102

[^16]: https://www.lewiscoleblog.com/spin-glass-models-2

[^17]: https://pubmed.ncbi.nlm.nih.gov/34021081/

[^18]: https://arxiv.org/abs/2102.11977

[^19]: https://www.ihes.fr/~ruelle/PUBLICATIONS/[90].pdf

[^20]: https://www.youtube.com/watch?v=i9EaYsR9cvA

[^21]: https://www.cirm-math.fr/RepOrga/2104/Slides/Baik.pdf

[^22]: https://link.aps.org/accepted/10.1103/PhysRevB.109.024431

[^23]: https://www.preprints.org/manuscript/202402.1058/v1

[^24]: https://hzshan.github.io/replica_method_in_SK_model.pdf

[^25]: https://www.math.toronto.edu/joaqsan/Resources/tomas_april2_slides.pdf

[^26]: https://indico.ictp.it/event/7607/session/348/contribution/1890/material/slides/0.pdf

[^27]: http://arxiv.org/pdf/1412.0170.pdf

[^28]: https://github.com/mcwitt/isg

[^29]: https://lewiscoleblog.com/spin-glass-models-4

[^30]: https://arxiv.org/pdf/1912.00793.pdf

[^31]: https://stackoverflow.com/questions/45399851/improving-python-code-in-monte-carlo-simulation/45403017

[^32]: https://ntrs.nasa.gov/api/citations/20140007519/downloads/20140007519.pdf

[^33]: https://github.com/mkan88/SGI

[^34]: https://developer.skao.int/projects/ska-sdp-resource-model/en/latest/_modules/ska_sdp_resource_model/simulate/monte_carlo.html

[^35]: https://www.g2qcomputing.com/QuantumMCMC.pdf

[^36]: https://sisl.readthedocs.io/en/v0.16.0/api/generated/sisl.physics.Overlap.html

[^37]: https://arxiv.org/abs/math/0604082

[^38]: https://arxiv.org/pdf/1905.03317.pdf

[^39]: https://pubs.aip.org/aip/jap/article/52/3/1697/503703/Faraday-rotation-measurements-of-time-dependent

[^40]: https://link.aps.org/accepted/10.1103/PhysRevB.96.054408

[^41]: https://www.numdam.org/item/AIHPB_2006__42_2_215_0/

[^42]: https://people.cas.uab.edu/~slstarr/lec1rev.pdf

[^43]: https://wt.iam.uni-bonn.de/fileadmin/WT/Inhalt/people/Anton_Bovier/lecture-notes/bovier-paris.pdf

[^44]: https://www.sciencedirect.com/science/article/abs/pii/0021999181900899

[^45]: https://web.stanford.edu/group/OTL/lagan/18012/SPINSUsageExample.pdf

[^46]: https://github.com/giselamarti/Exchange_MC

[^47]: https://www.jstor.org/stable/43736866

[^48]: http://arxiv.org/pdf/cond-mat/0609254.pdf

