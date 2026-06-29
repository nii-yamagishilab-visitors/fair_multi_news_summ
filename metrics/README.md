# Evaluation

A framework for evaluating the **fairness of LLM-generated multi-document news summaries**. Given summaries produced by a range of open-weight LLMs over news articles with known political lean (left / center / right), this pipeline measures how equitably each model represents its source documents.

---

## Pipeline

Run the steps below in order. Each step depends on the outputs of the previous one.

### 1. Generate summaries

Summaries must be generated first and placed under:

```
generated_summary/<method>/<data_type>/<model_name>/summaries.json
```

Supported **methods**: `all_input`, `balanced_input`, `debias_instruction`, `debias_persona`, `structured_prompt`, `oracle`

Supported **data types**: `random`, `lead_left`, `lead_right`, `lead_center`

---

### 2. Run fairness evaluation

Each metric family has its own subdirectory under `evaluation/` with a matching SLURM script and Python evaluation script.

#### Hard metrics

These measure structural source fairness — whether the summary draws on its input documents in a balanced way.

| Metric | Script | What it measures |
|---|---|---|
| `equal` | `eval_fairness_equal.sh` | Whether sentiment are distributed equally in the generated summaries
| `neutralisation` | `eval_fairness_neutralisation.sh` | Whether framing of summaries are neutralised
| `ratio` | `eval_fairness_ratio.sh` | Whether sentiment expressed in sentence proportions reflect source proportions

#### Soft metrics

These measure semantic and linguistic fairness properties.

| Metric | Script | What it measures |
|---|---|---|
| `entity_diversity` | `entity_diversity.sh` | Diversity of named entities drawn from each source |
| `entity_sentiment` | `entity_sentiment_diversity.sh` | Sentiment consistency per entity across sources |

Submit any evaluation job with:

```bash
sbatch evaluation/hard_metrics/equal/eval_fairness_equal.sh
sbatch evaluation/hard_metrics/neutralisation/eval_fairness_neutralisation.sh
sbatch evaluation/hard_metrics/ratio/eval_fairness_ratio.sh
sbatch evaluation/soft_metrics/entity_diversity/entity_diversity.sh
sbatch evaluation/soft_metrics/entity_sentiment/entity_sentiment_diversity.sh
```

Each script loops over all models and data types for the configured method. The active method is set via the METHOD variable at the top of each script (default: all_input). To run a different method, update METHOD before submitting. Results are written to:

```
# Hard metrics
evaluation/output/<method>/hard/<metric>/<data_type>/<model_name>/results.json

# Soft metrics
evaluation/output/<method>/soft/<metric>/<data_type>/<model_name>/results.json
```

> **Note:** `entity_sentiment` results are written under an additional `top_<k>/` level:
> ```
> evaluation/output/<method>/soft/entity_sentiment_diversity/top_<k>/<data_type>/<model_name>/results.json
> ```
> The default `TOP_K=2` can be changed at the top of `entity_sentiment_diversity.sh`.


---

## Project structure

```
evaluation/
├── hard_metrics/
│   ├── equal/
│   │   ├── eval_fairness_equal.sh
│   │   ├── fairness_sentence_level_equal.py
│   ├── neutralisation/
│   │   ├── eval_fairness_neutralisation.sh
│   │   ├── fairness_sentence_level_neutralisation.py
│   └── ratio/
│       ├── eval_fairness_ratio.sh
│       ├── fairness_doc_level.py
├── soft_metrics/
│   ├── entity_diversity/
│   │   ├── entity_diversity.py
│   │   ├── entity_diversity.sh
│   └── entity_sentiment/
│       ├── entity_sentiment_diversity.py
│       ├── entity_sentiment_diversity.sh
```

---

## Requirements

- Python 3.10+
- PyTorch with CUDA (required for GPU-based evaluation scripts)
- spaCy with the `en_core_web_sm` model (`python -m spacy download en_core_web_sm`)
- [NewsSentiment](https://github.com/fhamborg/NewsSentiment) (`pip install NewsSentiment`)
- A SLURM cluster with GPU (`qgpu`) and CPU (`qcpu`) partitions
- Model weights accessible via HuggingFace (Llama 3, Gemma 3, Qwen 2.5 families)

---

## Models evaluated

| Family | Variants |
|---|---|
| Llama 3 | 1B, 3B, 8B, 70B Instruct |
| Gemma 3 | 1B, 4B, 12B, 27B IT |
| Qwen 2.5 | 1.5B, 3B, 7B, 32B, 72B Instruct |