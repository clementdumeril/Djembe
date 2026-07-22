# Proposition : visualisation complexe des dununs

Objectif : remplacer l'explication Fourier actuelle, jugée trop abstraite avec
les flèches et le paramètre `n`, par une interface plus musicale :

- un cercle lisible à 12 subdivisions ;
- deux anneaux D1/D2 quand D2 existe ;
- une équation complexe LaTeX attachée à chaque rythme ;
- les coefficients affichés comme empreinte, sans modification dans un premier temps ;
- une carte vectorielle où chaque rythme devient un point.

## Idée centrale

Chaque rythme est représenté par un couple de fonctions D1/D2 et un vecteur
complexe de coefficients :

```text
R_piece(s) = [D1_piece(s), D2_piece(s)]
c_voice = (c1, c2, c3, c4, c5, c6)
```

avec :

```text
ck = (1 / A) · Σs as · exp(-i · k · θs)
```

où :

- `s` est une subdivision du cycle ;
- `θs = 2πs/12` ;
- `as` est la probabilité ou l'intensité de frappe sur cette subdivision ;
- `A = Σs as` normalise le motif ;
- `k` n'est plus présenté comme un nombre abstrait, mais comme une couche de
  structure rythmique.

Équation affichée :

```text
D_voice(s) = ρvoice/12 · [1 + 2 · Σk=1..5 |ck| · cos(2πks/12 + φk)
              + c6 · cos(πs)]
```

Les coefficients sont lus tels quels. Dans cette première version, on ne change
pas les valeurs : l'interface sert à comparer les équations des trois rythmes.

## Interface proposée

### 1. Cercle principal

Au centre :

- 12 positions fixes ;
- un anneau D1 ;
- un anneau D2 si la prise est appariée ;
- les frappes observées en points pleins sur deux anneaux lisibles.

Le cercle sert à lire les prises D1/D2 ; l'équation LaTeX porte la partie
mathématique.

### 2. Console d'équation

À droite :

```text
Rythme = c1·e¹ + c2·e² + c3·e³ + c4·e⁴ + c5·e⁵ + c6·e⁶
```

Chaque `ck` est affiché comme :

- amplitude `|ck|` ;
- phase `arg(ck)` ;
- nom musical de la couche.

Les noms affichés doivent être musicaux plutôt que techniques :

| Coefficient | Lecture intuitive |
|---|---|
| `c1` | gravité du cycle |
| `c2` | balancement en deux pôles |
| `c3` | triangle / trois appuis |
| `c4` | carré / quatre appuis |
| `c5` | asymétrie fine |
| `c6` | alternance rapide |

### 3. Carte vectorielle

En bas :

- chaque rythme devient un point dans l'espace des coefficients ;
- distance = différence de motif ;
- clic sur Manjanin/Maraka/Woloso pour charger son équation ;
- D1 et D2 sont calculés sur les mêmes prises appariées.

### 4. Mode retenu

Pour l'instant, un seul mode : **Lire**.

Il montre le motif réel, son équation, ses coefficients et sa position dans la
carte vectorielle. Les modes de composition ou d'édition restent volontairement
hors de cette version.

## Pourquoi c'est plus clair

L'utilisateur ne lit plus un `n` abstrait. Il lit des couches visibles :

- quand `c3` est fort, le cercle montre trois pôles ;
- quand `c6` est fort, l'alternance est plus serrée ;
- quand la phase est décalée, le motif pivote dans le cycle ;
- quand plusieurs coefficients se combinent, le cercle montre directement la
  forme résultante.

Le vocabulaire devient :

```text
un rythme = une forme sur le cercle = un vecteur complexe = une équation lisible
```

Nom retenu : **Boussole des rythmes**.

## Pourquoi seulement six coefficients ?

Le cycle est discrétisé en 12 subdivisions. Pour un signal réel sur 12 points,
la transformée de Fourier contient 12 coefficients, mais ils ne sont pas tous
indépendants :

- `c0` porte la moyenne, c'est-à-dire la densité globale du motif ;
- `c1..c5` ont chacun un miroir conjugué : `c11..c7` ;
- `c6` est le coefficient de Nyquist, l'alternance maximale possible sur une
  grille de 12.

Donc les couches indépendantes non triviales sont `c1..c6`. Afficher `c7..c11`
redonnerait la même information en miroir.
