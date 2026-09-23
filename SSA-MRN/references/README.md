# Official SSA-MRN source

- Repository: https://github.com/zhouchuanxu/SSA-MRN
- Paper: https://doi.org/10.1109/JSTARS.2025.3543827
- Pinned as a Git submodule at commit `a4ca40e407b12bf4c30f804384405ce321d11c51`.
- The upstream repository exposes the model definition (`network.py`) and a README pointing to PanCollection. It does not include the full training/evaluation pipeline, dataset configuration, or pretrained weights.
- Keep upstream files unchanged. Implement missing reproduction components under `src/ssamrn/` and `scripts/`, and track them in `docs/reproduction_status.md`.
