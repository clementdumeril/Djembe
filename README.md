# DJEMBE_LLM

Analyse de la musique de djembé malien à partir du corpus annoté **IEMP
Malian Jembe** (Polak, Tarsitani & Clayton, 2018).

Le chantier actif est la **V2 dununs** : on se concentre sur le dunun, le
tambour grave joué à la baguette, dans trois rythmes du corpus :
**Manjanin**, **Maraka** et **Woloso**.

## État du projet

Le projet est un prototype de recherche avancé :

- extraction des frappes de dunun ;
- alignement sur le cycle métrique ;
- analyse de motif, densité, tempo et micro-timing ;
- visualisations HTML autonomes avec audio embarqué ;
- caractérisation géométrique/Fourier des rythmes ;
- boussole des rythmes avec trois équations LaTeX D1/D2 comparables, courbes
  de modèle, boucle audio continue, transformations orthogonale/symétrique et
  synthétiseur repliable, catalogue combinatoire D1/D2 des variantes
  symétriques, comparaison interactive entre grille isochrone, mètre observé
  et micro-asynchronie D1/D2.

Les données brutes et sorties générées ne sont pas versionnées dans Git : elles
sont lourdes, régénérables, et liées à la licence du corpus.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Le projet a été vérifié localement avec Python 3.11.

## Pipeline V2

Lancer les notebooks depuis la racine du projet, dans cet ordre :

| Notebook | Rôle |
|---|---|
| `01_telechargement_V2.ipynb` | télécharge les pistes D1/D2 et annotations depuis OSF |
| `02_dunun_extraction_V2.ipynb` | aligne les frappes sur le cycle et génère `output/dunun_frappes_V2.csv` |
| `03_dunun_comparaison_V2.ipynb` | compare motif, densité, tempo, micro-timing et séparabilité |
| `04_dunun_visu_V2.ipynb` | génère la visualisation dynamique D1/D2 |
| `05_dunun_microtiming_V2.ipynb` | analyse précision, swing et asynchronie D1/D2 sur prises appariées |
| `06_dunun_microtiming_avance_V2.ipynb` | valide les moyennes par les distributions complètes et teste le lien micro-timing/intensité |
| `07_dunun_fourier_V2.ipynb` | empreinte de Fourier, équation complexe et distances entre rythmes |

Scripts principaux :

- `utils_V2.py` : téléchargement, chargement, extraction, couverture D1/D2 ;
- `dunun_viz_V2.py` : visualisation dynamique ;
- `boussole_rythmes_V2.py` : cercle D1/D2 + équations complexes LaTeX sur prises appariées.

Les notebooks constituent le **laboratoire scientifique** : ils explorent,
documentent et valident les méthodes. La Boussole est l'interface interactive
issue de ce laboratoire. Elle ne lance pas directement les notebooks dans le
navigateur : `boussole_rythmes_V2.py` reprend leurs calculs stabilisés et génère
une page autonome. Les notebooks 05–06 fondent la partie temporelle ; le
notebook 07 fonde les coefficients de Fourier et l'équation complexe.

## Point méthodologique important : D2

D2 n'est **pas disponible sur toutes les prises**. Ce n'est pas un bug : c'est
une asymétrie du corpus.

Règle retenue :

- les comparaisons globales entre rythmes utilisent **D1**, la voix complète ;
- les comparaisons **D1 vs D2** sont limitées aux prises réellement appariées ;
- aucune donnée D2 manquante n'est imputée.

Prises appariées observées localement :

| Rythme | Prises avec D1+D2 |
|---|---|
| Manjanin | 1, 4 |
| Maraka | 6 |
| Woloso | 2 |

Pour régénérer le rapport de couverture :

```bash
python qc_d2_coverage.py
```

Voir aussi [`docs/D2_COVERAGE.md`](docs/D2_COVERAGE.md).

## Vérification rapide

```bash
python check_project.py
python prepare_git.py
python boussole_rythmes_V2.py
```

## Résultats déjà observés

- Le motif du dunun distingue fortement les trois rythmes.
- Woloso alterne deux sous-motifs : cycles `.1` aux subdivisions 1, 3, 6, 9, 12
  et cycles `.2` aux subdivisions 1, 4, 7, 10.
- Le micro-timing est stable et précis : la richesse du dunun est dans le
  placement temporel, pas dans une classification de types de frappe.
- Le dunun n'a pas de typage claqué/tonique/basse comme le djembé : on étudie
  surtout **où** et **quand** il frappe.

## Structure

```text
DJEMBE_LLM/
├── 01_..._V2.ipynb à 07_..._V2.ipynb
├── utils_V2.py
├── dunun_viz_V2.py
├── boussole_rythmes_V2.py
├── check_project.py
├── prepare_git.py
├── qc_d2_coverage.py
├── docs/
├── corpus/     # données brutes locales, ignorées par Git
├── output/     # sorties générées, ignorées par Git
└── archive/    # anciens essais et V1, audio lourd ignoré par Git
```

## Données et licence

Corpus : **IEMP Malian Jembe** (Polak, Tarsitani & Clayton, 2018), OSF
`M652X`, licence **CC BY-NC-ND 4.0**.

Usage recommandé : recherche, pédagogie et exploration personnelle, avec citation
de la source.
