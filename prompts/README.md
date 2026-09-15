# Prompts

Prompt templates used for the generative baselines reported in the paper.

| File | Purpose | Paper reference |
|---|---|---|
| `gpt4_baseline.yaml` | GPT-4 templates, zero-shot and 5-shot, with decoding and exemplar settings | Appendix A; Tables 6, 12 |

The YAML is the single copy of the prompt text — deliberately not duplicated in a
second human-readable file, so the two cannot drift apart.

## Scope and status

These templates are transcribed verbatim from **Appendix A** of the manuscript and
are the exact prompts used to produce the GPT-4 rows in Table 6 and the
head-to-head preferences in Table 12.

**What is provided:** the prompt text, the model snapshot identifier, and the
exemplar-sampling rule.

**What was not recorded:** GPT-4 decoding parameters (temperature, maximum output
tokens) and the seeding of the exemplar draw. The manuscript does not report them
and the original API scripts were not retained, so these fields are `null` in the
YAML rather than filled with guessed defaults.

**What is not provided:** a runnable OpenAI API driver. `training_evaluation/baseline_comparison.py`
names GPT-4 as a baseline but does not implement the API calls. Anyone re-running
this baseline must write the API loop themselves against these templates. This gap
is also listed in the root [`README.md`](../README.md).

## Model snapshot

The baseline uses **`gpt-4-0613`**, the snapshot frozen when the evaluation protocol
was fixed. As stated in Section 5.3 of the paper, this is reported as a **historical
reference point, not a current state-of-the-art comparator**. Re-running these
prompts against a newer model will not reproduce the reported numbers and should
not be described as a reproduction of them.

## Exemplar selection (5-shot)

The five in-context exemplars were drawn at random from the **training partition
(2015–2022)** and are disjoint from the **2023–2024 test period**, so the few-shot
baseline introduces no test-set leakage.

Neither the seed used for the random draw nor the identity of the five documents
selected was recorded. Re-sampling under the same rule reproduces the procedure, not
the specific exemplars.
