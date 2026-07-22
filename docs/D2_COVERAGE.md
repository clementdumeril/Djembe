# Couverture D2

Dans le corpus V2, D2 n'est pas disponible sur toutes les prises. Ce n'est pas
une erreur du pipeline : c'est une propriété du corpus local.

Règle méthodologique retenue :

- les comparaisons globales entre rythmes utilisent D1, la voix complète ;
- les comparaisons D1/D2 ne sont calculées que sur les prises appariées ;
- aucune frappe D2 manquante n'est imputée ou synthétisée.

Prises appariées observées localement :

| Rythme | Prises avec D1+D2 |
|---|---|
| Manjanin | 1, 4 |
| Maraka | 6 |
| Woloso | 2 |

Pour régénérer les rapports locaux :

```bash
python qc_d2_coverage.py
```

Sorties générées :

- `output/d2_coverage_V2.csv`
- `output/d2_pair_metrics_V2.csv`
