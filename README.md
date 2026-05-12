# Optimizer-Induced Mode Connectivity: From AdamW to Muon

This is the codebase for result reproducibility for our manuscript: [Optimizer-Induced Mode Connectivity: From AdamW to Muon](https://arxiv.org/abs/2605.09991)

## Setup
1. `conda create -n mc python=3.11` and `conda activate mc`
2. `pip install -r requirements.txt`
3. run `pip install git+https://github.com/KellerJordan/Muon`
4. create datasets following [dataset_setup.txt](dataset_setup.txt). The code can be found in [data_prepare](data_prepare).
   
We've added [environment.yml](environment.yml) as a backup if the requirements installation goes wrong.

## Same-Optimizer Connectivity
<div align="center">
  <img src="figs/table1.png" width="700"/>
</div>

To reproduce Table 1's results, train with [run_train.sh](intra_mc/run_train.sh), then merge with [run_merge.sh](intra_mc/run_merge.sh), finally run eval with [run_eval.sh](intra_mc/run_eval.sh). This will give all evaluation losses for all datasets and seeds, for exact mean and std number computation, feed the resulted file `loss_interp_mean_err.json` into [compute_mc_barrier.py](compute_mc_barrier.py).

## Spectral Phase Transition

<div align="center">
  <img src="figs/intra_srank.png" width="800"/>
  <img src="figs/inter_srank_adam.png" width="800"/>
</div>

Train with [run_train.sh](spectral_probe/run_train.sh), then merge [run_merge.sh](spectral_probe/run_merge.sh), then extract singular value histograms for models along the interpolating path and generate final plots by [run_eval.sh](spectral_probe/run_eval.sh) to plot for fixed layer and fixed weight. 

## Out-of-Distribution Generalization
<div align="center">
  <img src="figs/ood_main.png" width="800"/>
</div>

Train with [run_train.sh](ood/run_train.sh), then merge with [run_merge.sh](ood/run_merge.sh), finally generate OOD plots with [run_eval.sh](ood/run_eval.sh).

## Citation

```bibtex
@misc{zhang2026optimizerinducedmodeconnectivityadamw,
      title={Optimizer-Induced Mode Connectivity: From AdamW to Muon}, 
      author={Fangzhao Zhang and Sungyoon Kim and Erica Zhang and Yiqi Jiang and Mert Pilanci},
      year={2026},
      eprint={2605.09991},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.09991}, 
}
```
For any question related to this code repo, please contact zfzhao@stanford.edu.
