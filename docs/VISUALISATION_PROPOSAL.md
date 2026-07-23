# Proposition : visualisation complexe des dununs

Objectif : remplacer l'explication Fourier actuelle, jugée trop abstraite avec
les flèches et le paramètre `n`, par une interface plus musicale :

- un cercle lisible à 12 subdivisions ;
- deux anneaux D1/D2 quand D2 existe ;
- une équation complexe LaTeX attachée à chaque rythme ;
- les coefficients affichés comme empreinte, puis modifiables dans un mode synthé ;
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
- `k` indique le nombre d'oscillations de l'onde complexe dans le cycle.

On peut aussi l'écrire avec les racines douzièmes de l'unité :

```text
zs = exp(2πi · s/12)
ck = Σs xs · zs^(-k)
```

Les `zs` sont les 12 points régulièrement posés sur le cercle complexe. Chaque
coefficient compare le motif complet à une onde qui tourne autour de ce cercle.

Équation affichée :

```text
D_voice(s) = ρvoice/12 · [1 + 2 · Σk=1..5 |ck| · cos(2πks/12 + φk)
              + c6 · cos(πs)]
```

Les coefficients peuvent être lus tels quels en mode **Original**, ou modifiés
en mode **Synthé** pour reconstruire une nouvelle courbe jouable.

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
- nom mathématique de l'onde.

Les noms affichés restent techniques pour éviter les métaphores ambiguës :

| Coefficient | Lecture intuitive |
|---|---|
| `c1` | 1 oscillation par cycle |
| `c2` | 2 oscillations par cycle |
| `c3` | 3 oscillations par cycle |
| `c4` | 4 oscillations par cycle |
| `c5` | 5 oscillations par cycle |
| `c6` | alternance maximale sur 12 subdivisions |

### 3. Carte vectorielle

En bas :

- chaque rythme devient un point dans l'espace des coefficients ;
- distance = différence de motif ;
- clic sur Manjanin/Maraka/Woloso pour charger son équation ;
- D1 et D2 sont calculés sur les mêmes prises appariées.

### 4. Modes retenus

Deux modes simples :

- **Original** : voir le motif réel, son équation, ses coefficients et sa
  position dans la carte vectorielle ;
- **Synthé** : modifier la densité, les amplitudes `|ck|` et les phases, puis
  jouer la courbe reconstruite.

## Pourquoi c'est plus clair

L'utilisateur ne lit plus un `n` abstrait. Il lit des ondes visibles :

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

Donc les ondes indépendantes non triviales sont `c1..c6`. Afficher `c7..c11`
redonnerait la même information en miroir.

Important : `c1..c6` ne sont pas les six premières subdivisions du rythme. Ce
sont six ondes de forme. Chaque onde regarde les 12 subdivisions à la fois :
elle mesure une manière de se répartir dans le cycle.

Le mot "couche" signifie donc "onde test". On ne découpe pas le rythme en six
tranches temporelles. On le compare à six horloges :

- `c1` : une horloge qui fait 1 tour pendant le cycle ;
- `c2` : une horloge qui fait 2 tours ;
- `c3` : une horloge qui fait 3 tours ;
- etc.

Si les frappes tombent dans le même sens que l'onde, les vecteurs complexes
s'additionnent et `|ck|` devient grand. Si elles sont réparties dans des sens
opposés, les vecteurs s'annulent et `|ck|` devient petit.

## Analyse des coefficients

Pour chaque voix D1/D2 :

- l'amplitude `|ck|` mesure la force de l'onde `k` ;
- la phase `arg(ck)` mesure où cette onde est placée dans le cycle ;
- la combinaison des six ondes reconstruit la courbe du rythme ;
- comparer deux rythmes revient à comparer leurs vecteurs de coefficients.

La carte des signatures projette ces vecteurs dans un plan :

- axe x = composante principale 1, première direction qui résume les distances
  entre signatures ;
- axe y = composante principale 2, deuxième direction qui résume les distances
  restantes ;
- ce ne sont pas des axes musicaux fixes, mais des axes statistiques calculés
  à partir des rythmes présents dans la page.

## Forme courbe

La page affiche aussi une courbe par voix :

```text
x = subdivision Polak du cycle, de 1 à 12
y = probabilité moyenne de frappe
```

Les points sont les observations. La courbe est le modèle reconstruit à partir
des coefficients complexes.

## Synthétiseur de coefficients

Le synthétiseur transforme les coefficients en paramètres de composition :

- `ρ` règle la densité moyenne du motif ;
- `|ck|` règle la force de l'onde `k` ;
- `arg(ck)` décale cette onde dans le cycle ;
- `c6` règle l'alternance maximale, qui doit rester réelle pour conserver un
  motif réel.

La courbe reconstruite peut devenir négative si les coefficients sont poussés
fortement. Pour l'écoute, la page coupe ces valeurs à zéro avant de déclencher
les frappes.

L'interface est organisée comme un instrument : cercle métrique à gauche,
synthétiseur à droite, puis courbes temporelles en dessous. La lecture fonctionne
en boucle continue. Un ordonnanceur audio ne prépare qu'une courte fenêtre
d'avance ; une modification de `ρ`, `|ck|` ou `arg(ck)` est donc prise en compte
sur la prochaine subdivision non encore programmée, sans relancer le cycle.

## Transformations automatiques

Deux opérations construisent automatiquement une nouvelle voix :

- **projection orthogonale** : pour la voix éditée `x` et l'autre voix `y`, on
  retire d'abord leur moyenne, puis on calcule
  `q = xc - <xc,yc>/<yc,yc> yc`. Le résultat est donc orthogonal à l'autre voix
  dans l'espace centré des 12 subdivisions. Une constante et un facteur
  d'échelle le rendent positif et conservent sa densité sans modifier cette
  orthogonalité centrée ;
- **symétrie miroir** : la voix est réfléchie autour d'un axe passant par deux
  subdivisions opposées. Pour un axe d'indice `a`,
  `Dsym(s) = D((2a-s) mod 12)`.

Les deux opérations sont reconverties en coefficients complexes puis restent
modifiables avec les curseurs.

## Étudier l'impact du micro-timing

Le motif répond à **où** une frappe existe dans les 12 subdivisions. Le
micro-timing répond à **quand**, en millisecondes, elle tombe autour de cette
position. Pour chaque voix et subdivision, la page calcule sur les prises
appariées :

- l'écart moyen à la grille
  `δs = moyenne[(rel - (s-1)/12) × durée_cycle × 1000]` ;
- sa dispersion, qui mesure la précision du placement ;
- la précision résiduelle après retrait du placement moyen de chaque
  subdivision.

Dans le graphique, le cercle vide et sa barre représentent la moyenne observée
et sa dispersion. Le point plein indique le placement réellement envoyé au
moteur audio. Le mode **Grille** impose `δs = 0`. Le mode **Feel observé** joue
`tfrappe = tgrille + αδs`, où le curseur d'impact `α` varie de 0 à 2. On peut
ainsi comparer le même motif et les mêmes amplitudes avec ou sans micro-timing,
sans confondre cet effet avec une modification des coefficients. Le décalage
audio est plafonné à ±60 ms pour éviter qu'une frappe ne traverse la
subdivision voisine quand l'effet est exagéré.
