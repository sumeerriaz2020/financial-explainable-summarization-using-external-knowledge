# Configuration files

Every configuration used to produce a reported result, named so it can be mapped
back to the table or figure it corresponds to.

## Config → paper mapping

| Config file | Produces | Paper reference |
|---|---|---|
| `model_config.yaml` | Architecture for all proposed-system runs | §3.3, §3.6.2, Table 1 |
| `training_config.yaml` | The reported training run | Tables 3, 4, 5; Table 7 final row |
| `baselines.yaml` | All six baselines | §3.7.3, Tables 2, 6, 12 |
| `ablation.yaml` | Sequential-addition ablation + KG integration strategies | Tables 7, 8; Figure 5 |
| `evaluation.yaml` | Metric definitions, sample sizes, statistical protocol | §3.7, Tables 11, 14, 16; Figure 6 |
| `fibo_modules.yaml` | FIBO module selection and extension mapping | §3.2 |

## Values

All values here are traceable to the manuscript; [`docs/PAPER_FACTS.md`](../docs/PAPER_FACTS.md)
collects every reported number. Values tagged `[not in paper]` are implementation
defaults, not reported settings.

Validation and test figures are kept separate in `training_config.yaml`: ROUGE-L
0.487 and BERTScore 0.686 are the validation figures used for model selection
(Section 3.6.2); the reported test figures are 0.497 and 0.673 (Table 3).

The HMM regime detector, the Page-Hinkley drift test (Section 3.4.5) and the
REINFORCE-style MESA update (Section 3.4.1) are recorded as parameter values only;
the mechanisms are not included in this release.
