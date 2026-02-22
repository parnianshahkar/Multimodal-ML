# Shapley-fusion

Welcome!

This project presents a new fusion method, **Shapley-fusion**, that encourages **cross-modal diversity** in multimodal ML and promotes **complementary modalities**, while each modality learns its **own independent model**.

## Overview

Shapley-fusion is designed to:
- Encourage *diversity* across modalities.
- Encourage *complementarity*.
- Keep *independent modality models* while learning an effective fusion strategy.

If you want the quickest “gist” of what’s new, start here:

- `train_one_epoch_shapley` in `meta_fusion/methodsextra.py`

## Origin / Attribution

This repository is **mainly based on** the `meta_fusion` codebase:

- Upstream project: **meta_fusion** — https://github.com/ZiyiLiang/fusion/tree/main  
- Author: ZiyiLiang and contributors

### What changed vs upstream

- `experiments_real/` is **unchanged** from upstream.
- Most other folders include **additional and/or modified code** to implement and run **Shapley-fusion**.

## Repository structure (high-level)

- `experiments_real/` — unchanged from upstream `meta_fusion`
- `meta_fusion/` — includes Shapley-fusion additions (see `methodsextra.py`)
- Other folders — modified/extended for Shapley-fusion experiments and utilities

## License

This project includes code derived from the upstream repository. Please ensure you comply with the upstream license and retain required notices.
