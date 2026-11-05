# SKIM: Pruning Large Language Model Agents via Selective Knowledge Informed Masking

Code for the NeurIPS 2026 paper. Project page: https://moon4sake.github.io/skim

## Quick start

```bash
pip install -e .
./run.sh prune qwen35_2b 0.5    # SKIM pruning at 50% sparsity
./run.sh kd    qwen35_2b 0.5    # SKIM distillation
./run.sh eval  qwen35_2b 0.5
```

Datasets are downloaded to `data/` on first use.

## Citation

```bibtex
@inproceedings{choi2026skim,
  title     = {{SKIM}: Pruning Large Language Model Agents via Selective Knowledge Informed Masking},
  author    = {Choi, Moonseok and Jeong, Jongwon and Kang, Minki and Nam, Giung and Lee, Juho},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT
