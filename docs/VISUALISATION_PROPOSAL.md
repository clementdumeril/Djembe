# Proposition : visualisation complexe des dununs

Objectif : remplacer l'explication Fourier actuelle, jugée trop abstraite avec
les flèches et le paramètre `n`, par une interface plus musicale :

- un cercle lisible à 12 subdivisions ;
- deux anneaux D1/D2 quand D2 existe ;
- une équation complexe attachée au rythme ;
- des coefficients manipulables qui modifient la forme du motif en direct ;
- une carte vectorielle où chaque rythme devient un point.

## Idée centrale

Chaque rythme est représenté par un vecteur complexe de coefficients :

```text
R = (c1, c2, c3, c4, c5, c6)
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

Reconstruction interactive :

```text
R̂(θ) = c0 + 2 · Re(Σk wk · ck · exp(i · k · θ))
```

`wk` est un curseur utilisateur : il permet d'écouter/voir ce que chaque couche
apporte au rythme.

## Interface proposée

### 1. Cercle principal

Au centre :

- 12 positions fixes ;
- un anneau D1 ;
- un anneau D2 si la prise est appariée ;
- les frappes réelles en points pleins ;
- la reconstruction mathématique en halo ou courbe douce.

Quand on bouge un coefficient, le halo change immédiatement.

### 2. Console d'équation

À droite :

```text
Rythme = c1·e¹ + c2·e² + c3·e³ + c4·e⁴ + c5·e⁵ + c6·e⁶
```

Chaque `ck` est affiché comme :

- amplitude `|ck|` ;
- phase `arg(ck)` ;
- curseur de poids `wk`.

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
- D1 et D2 peuvent être affichés séparément, mais D2 seulement sur les prises
  appariées.

### 4. Modes

Trois modes simples :

1. **Lire** : voir le motif réel, sans math.
2. **Décomposer** : activer/désactiver les coefficients.
3. **Composer** : bouger les coefficients et créer un motif reconstruit.

## Pourquoi c'est plus clair

L'utilisateur ne manipule plus un `n` abstrait. Il manipule des couches visibles :

- quand `c3` monte, le cercle montre trois pôles ;
- quand `c6` monte, l'alternance devient plus serrée ;
- quand la phase tourne, le motif pivote dans le cycle ;
- quand plusieurs coefficients se combinent, le cercle montre directement la
  forme résultante.

Le vocabulaire devient :

```text
un rythme = une forme sur le cercle = un vecteur complexe = une équation réglable
```

Nom de travail : **Boussole complexe des dununs**.
