Teacher: `teacher_000032768000`, held-out success 98%; success during collection 100%. Student success: mean ± std over training seeds (pooled Wilson 95% CI).

| Condition | Number of demonstrations | Teacher success rate | Student success rate | Evaluation episodes |
|---|---:|---:|---:|---:|
| Baseline | 100 | 98% | 69.3% ± 4.0% [0.64, 0.74] (3 seeds) | 100 per seed |
| Ablation | 50 | 98% | 39.0% ± 8.9% [0.34, 0.45] (3 seeds) | 100 per seed |
| Ablation | 25 | 98% | 13.7% ± 5.5% [0.10, 0.18] (3 seeds) | 100 per seed |
| Reference: teacher_000032768000 | – | – | 98.0% | 100 |
| Reference: random_action | – | – | 0.0% | 100 |
| Reference: zero_action | – | – | 0.0% | 100 |
