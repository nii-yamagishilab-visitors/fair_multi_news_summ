# When Bigger Isn't Better: A Comprehensive Fairness Evaluation of Political Bias in Multi-News Summarisation

> Huang et al.
> [[Paper]](https://aclanthology.org/2026.acl-long.894/)

We study the fairness of LLM-generated summaries over politically diverse multi-document news corpora. The repository is organised into two main modules: dataset construction and fairness evaluation. This repository provides the code for constructing the **FairNews** dataset and evaluating political fairness using five fairness metrics.

---

## Repository structure

```
.
├── process_all_the_news/   # Dataset construction pipeline
└── metrics/                # Fairness evaluation pipeline
```

---

## Modules

### 1. `process_all_the_news/`

Scripts for constructing the multi-document news dataset from All The News, including political bias labelling, event grouping, and dataset filtering.

See [`process_all_the_news/README.md`](process_all_the_news/README.md) for full usage instructions.

### 2. `metrics/`

Scripts for evaluating the fairness of LLM-generated summaries across five metrics (equal distribution, neutralisation, ratio, entity diversity, and entity sentiment diversity), covering 13 models from the Llama 3, Gemma 3, and Qwen 2.5 families.

See [`metrics/README.md`](metrics/README.md) for full usage instructions.

---

## Citation

If you use this code or dataset in your work, please cite:

```bibtex
@inproceedings{huang2026bigger,
  title={When Bigger Isn’t Better: A Comprehensive Fairness Evaluation of Political Bias in Multi-News Summarisation},
  author={Huang, Nannan and Maab, Iffat and Yamagishi, Junichi},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={19532--19563},
  year={2026}
}
```
