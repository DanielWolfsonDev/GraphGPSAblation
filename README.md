### README

This repository contains the main code for the ablation study, built upon the GraphGPS GitHub repository. To facilitate transparency, a git diff file is included, highlighting all modifications made to the original GraphGPS code.

Additionally, the repository provides a script named analyze_weights, which implements the interpretability component of the project.

The included xlsx file, Ablation Details, offers comprehensive information, including intermediate results from the ablation algorithms.
### Installation

The installation is based on GraphGPS - we did not include any additional packages.

### Running the Experiments

1. Running the main script:

To run unablated experiment, run original GraphGPS scripts. For example:

```bash
python main.py --cfg configs/GPS/peptides-func-GPS.yaml  --repeat 10  wandb.use True
```

To run ablated experiemnts, add the configuration flags: "gt.mpnn_ablation_list" and "gt.attention_ablation_list".
For example, to run ablated experiment which removes the first MPNN sublayer and second and third attention sublayers of the 
Peptides-Func model, run:
```bash
python main.py --cfg configs/GPS/peptides-func-GPS.yaml  --repeat 10  wandb.use True gt.mpnn_ablation_list [0] gt.attention_ablation_list [1,2]
```

2. Extracting interpretability results:

First, run the main script with the flag "train.mode" set to "log-attn-weights":
```bash
python main.py --cfg configs/GPS/peptides-func-GPS.yaml  wandb.use True train.mode log-attn-weights  
```
This script generate a "PT" file.

Then, run the "analyze_weights" script with the "PT" file as a parameter.
The script will print detailed interpretability results per layer.
```bash
python analyze_weights.py <PT file>  
```