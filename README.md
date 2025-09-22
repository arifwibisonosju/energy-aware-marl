# energy-aware-marl
Simulation and implementation of Multi-Agent Reinforcement Learning (MARL) algorithms for coordinating Autonomous Underwater Vehicles (AUVs) in Internet of Underwater Things (IoUT) scenarios with an emphasis on energy efficiency.
This repository contains training and evaluation scripts for a multi-AUV coordination system using MARL (Multi-Agent Reinforcement Learning) algorithms, including MADDPG, MAPPO, and MODDPG.

## File Structure

| File / Folder                    | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `train_multi_agent.py`          | Initial version of MADDPG, includes model saving to `saved_models/maddpg/` |
| `train_mappo.py`                | MAPPO implementation, revised and shows simulation                         |
| `train_moddpg.py`               | MODDPG implementation, revised and shows simulation                         |
| `train_multi_agent_maddpg.py`   | Deprecated version (poor performance), kept for reference                   |
| `eval_multi_general.py`         | General evaluation script, use `--algo` and `--model_path` arguments        |
| `evaluate_baselines.py`         | Additional evaluation script for comparing baselines                        |
| `saved_models/`                 | Automatically stores trained models (subfolders: `maddpg/`, `mappo/`, `moddpg/`) |

---

## How to Train

### 1. MADDPG (initial version)
```bash
python train_multi_agent.py
```

### 2. MAPPO
```bash
python train_mappo.py
```

### 3. MODDPG
```bash
python train_moddpg.py
```

> 💡 Training outputs are saved under `saved_models/{algo_name}/`.

---

## How to Evaluate

### Evaluate specific model:
```bash
python eval_multi_general.py --algo mappo --model_path saved_models/mappo
```

### Or use default configuration:
```bash
python eval_multi_general.py
```

### Evaluate baselines:
```bash
python evaluate_baselines.py
```

---

## Notes

- `train_multi_agent_maddpg.py` is not used due to unstable performance.
- Both MAPPO and MODDPG include simulation visualization.
- Ensure dependencies are properly installed for your environment.

---

## Model Output Directory

```
saved_models/
├── maddpg/
├── mappo/
└── moddpg/
```
## Citation

If this repository helps you in your academic research, you are encouraged to cite our paper.  
Here is an example BibTeX entry:

```python
@ARTICLE{Wibisono2025MARL,
  author={Wibisono, Arif and Song, Hyoung-Kyu and Lee, Byung Moo},
  journal={IEEE Access},  
  title={Energy-Aware MARL for Coordinated Data Collection in Multi-AUV Systems},  
  year={2025},  
  volume={13},  
  pages={155835-155854},  
  issn={2169-3536},  
  doi={10.1109/ACCESS.2025.3606016}
}
```

