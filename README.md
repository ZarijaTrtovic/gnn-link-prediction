# Predviđanje veza u grafovima korišćenjem grafovskih neuronskih mreža

Projekat iz predmeta **Računarska inteligencija** na 4. godini, smer Informatika, **Matematički fakultet Univerziteta u Beogradu**.

**Autori:**
- Zarija Trtović
- Bogdan Delić

## O projektu

Ovaj projekat se bavi razvojem i poređenjem metoda za **predviđanje veza (engl. link prediction)** u grafovima, sa primenom na citatne mreže. Implementirana su tri pristupa, sva od nule:

1. **Common Neighbors (CN)** — heuristički baseline algoritam
2. **Graph Autoencoder (GAE)** — neuronski model zasnovan na grafovskim konvolucijama
3. **Variational Graph Autoencoder (VGAE)** — varijaciona generalizacija GAE modela

Cilj je da se, na osnovu strukture postojećeg grafa i atributa čvorova, predvidi koji parovi čvorova bi trebalo da budu povezani, što je tipičan zadatak u analizi citatnih mreža, društvenih mreža i bioinformatici.

U skladu sa zahtevima predmeta, **kompletne implementacije algoritama su napisane od nule** — nisu korišćeni gotovi GAE/VGAE moduli (npr. `torch_geometric.nn.GAE`), već se sve enkodiranje, dekodiranje, uzorkovanje i funkcije gubitka implementiraju ručno. Korišćeni su samo bazični grafovski konvolucioni slojevi (`GCNConv`) kao gradivni blokovi.

## Skupovi podataka

Korišćena su dva standardna skupa podataka iz Planetoid kolekcije za rad sa citatnim grafovima:

| Skup     | Čvorovi | Grane  | Dim. atributa | Klase |
|----------|---------|--------|---------------|-------|
| Cora     | 2.708   | 10.556 | 1.433         | 7     |
| CiteSeer | 3.327   | 9.104  | 3.703         | 6     |

Čvorovi predstavljaju naučne radove, grane citate među radovima, a atributi čvora predstavljaju bag-of-words reprezentaciju sadržaja rada.

## Struktura repozitorijuma

```
gnn-link-prediction/
├── notebooks/
│   ├── data_exploration.ipynb     # eksplorativna analiza Cora skupa
│   ├── citeseer_data.ipynb        # eksplorativna analiza CiteSeer skupa
│   ├── GAE.ipynb                  # GAE na Cora
│   ├── GAE_citeseer.ipynb         # GAE na CiteSeer
│   ├── VGAE.ipynb                 # VGAE na Cora
│   ├── VGAE_citeseer.ipynb        # VGAE na CiteSeer
│   └── CommonNeighbors.ipynb      # baseline na oba skupa
├── data/                          # mesto za sirove podatke (Cora, CiteSeer)
├── results/                       # generisane slike i grafici
└── README.md
```

## Implementirani algoritmi

### Common Neighbors (baseline)

Klasičan heuristički metod koji rangira parove čvorova prema broju zajedničkih suseda:

$$\text{CN}(u, v) = |N(u) \cap N(v)|$$

Ne zahteva nikakvu obuku. Služi kao referentna tačka u odnosu na koju se mere neuronski modeli.

### Graph Autoencoder (GAE)

Enkoder-dekoder arhitektura zasnovana na grafovskim konvolucijama (Kipf & Welling, 2016):

- **Enkoder:** dvoslojna grafovska konvoluciona mreža (GCN)
  - `GCNConv(in → hidden) + ReLU`
  - `GCNConv(hidden → out)` — daje latentne reprezentacije čvorova `Z`
- **Dekoder:** rekonstrukcija matrice susedstva preko unutrašnjeg proizvoda
  - $\hat{A}_{uv} = \sigma(z_u^\top z_v)$
- **Funkcija gubitka:** binarna unakrsna entropija (`BCEWithLogitsLoss`) nad pozitivnim i negativno odabranim (engl. *negative sampling*) granama.

### Variational Graph Autoencoder (VGAE)

Varijaciona generalizacija GAE-a koja čvor reprezentuje kao raspodelu, a ne kao tačku:

- **Enkoder:** zajednički GCN sloj, zatim dve paralelne GCN glave koje daju:
  - $\mu$ — srednja vrednost latentne raspodele
  - $\log \sigma$ — log standardna devijacija
- **Reparametrizacija:** $z = \mu + \varepsilon \cdot \exp(\log \sigma), \quad \varepsilon \sim \mathcal{N}(0, I)$
- **Dekoder:** isti kao kod GAE-a (unutrašnji proizvod)
- **Funkcija gubitka:** rekonstrukcioni gubitak + KL divergencija u odnosu na $\mathcal{N}(0, I)$:

$$\mathcal{L} = \mathcal{L}_{\text{rec}} + \beta \cdot \text{KL}\!\left[q(Z|X,A) \;\|\; p(Z)\right]$$

gde je $\beta = 0.005$ težina KL člana.

## Hiperparametri

Najbolje konfiguracije pronađene pretragom po mreži:

| Model | Skup     | Skriveno | Izlaz | Lr    | Epohe |
|-------|----------|---------:|------:|------:|------:|
| GAE   | Cora     | 128      | 64    | 0.01  | 200   |
| GAE   | CiteSeer | 64       | 32    | 0.01  | 200   |
| VGAE  | Cora     | 128      | 64    | 0.01  | 300   |
| VGAE  | CiteSeer | 128      | 64    | 0.01  | 300   |

Optimizator: **Adam**. Negativno uzorkovanje je balansirano sa brojem pozitivnih grana.

## Rezultati

Evaluacija je vršena pomoću metrika **AUC** (površina ispod ROC krive) i **AP** (prosečna preciznost) nad test skupom grana.

### Cora

| Model            | AUC    | AP     |
|------------------|-------:|-------:|
| Common Neighbors | 0.6930 | 0.6891 |
| GAE              | 0.9200 | 0.9296 |
| VGAE             | 0.9133 | 0.9240 |

### CiteSeer

| Model            | AUC    | AP     |
|------------------|-------:|-------:|
| Common Neighbors | 0.6714 | 0.6714 |
| GAE              | 0.8966 | 0.9008 |
| VGAE             | 0.8984 | 0.9049 |

Oba neuronska modela značajno nadmašuju heuristički baseline, što je očekivano s obzirom na to da koriste i atribute čvorova, a ne samo strukturu grafa. GAE i VGAE postižu uporedive rezultate, pri čemu VGAE blago dobija na CiteSeer-u.

## Vizualizacije

U direktorijumu `results/` nalaze se generisani grafici i vizualizacije:

- vizualizacija podgrafova Cora i CiteSeer skupova (`*_visualization.png`)
- distribucije stepena čvorova (`*degree_distribution.png`)
- krive gubitka, AUC i AP po epohama (`*_loss_curve.png`, `*_auc_curve.png`, `*_ap_curve.png`)
- poređenje hiperparametarskih konfiguracija (`*_hyperparams.png`)
- t-SNE projekcije polaznih atributa čvorova i naučenih embedding-a (`*_tSne_feature*.png`, `*_tSne_embedding*.png`)
- ROC kriva za Common Neighbors (`cn_roc_curve.png`)

## Pokretanje

Projekat je realizovan kroz Jupyter notebook-ove. Za pokretanje su potrebni:

- Python 3.x
- PyTorch
- PyTorch Geometric (za učitavanje skupova i `GCNConv` sloj)
- NumPy, scikit-learn (metrike), Matplotlib, NetworkX

Skupovi podataka se automatski preuzimaju preko `torch_geometric.datasets.Planetoid` pri prvom pokretanju i smeštaju u `data/`.

Preporučeni redosled prolaska kroz notebook-ove:

1. `data_exploration.ipynb` i `citeseer_data.ipynb` — upoznavanje sa podacima
2. `CommonNeighbors.ipynb` — baseline
3. `GAE.ipynb` i `GAE_citeseer.ipynb` — neuronski model bez varijacionog dela
4. `VGAE.ipynb` i `VGAE_citeseer.ipynb` — varijaciona varijanta
