# NaviFormer
[![](images/python.svg)](https://www.python.org)
[![](images/framework.svg)](https://pytorch.org)
[![](images/license.svg)](LICENSE)
[![](images/publication.svg)](https://neurips.cc)

This repository is the official implementation of the following paper:

> Daniel Fuertes, Andrea Cavallaro, Carlos R. del-Blanco, Fernando Jaureguizar, Narciso García, "NaviFormer: A Deep Reinforcement Learning Transformer-like Model to Holistically Solve the Navigation Problem", submitted to NeurIPS, 2024.

If you find this repository useful for your work, please cite our paper:

```
<Citation> - (Under review)
```

## Installation
The code has been evaluated under the following dependencies:
* Ubuntu 22.04.4 LTS
* nvidia-drivers == 535.161.08
* python3 == 3.10.12
* python3-venv == 3.10.12

To install the code, clone this repository and go to the working directory. Then, create a virtual environment and install the requirements as follows:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd benchmarks/nop/methods
git clone --recursive https://github.com/omron-sinicx/neural-astar
pip install neural-astar/.
cd ../../..
```

## Reproducibility
To reproduce the results of the paper, please download and unzip the datasets and pretrained weights from [here](). Place the datasets' folder (`data`) and the pretrained weights' folder (`pretrained`) in the working directory and run the following scripts:

```bash
source replicate_results.sh
source replicate_ablation.sh
source replicate_comparison.sh
```

## General usage

First, it is necessary to create test and validation sets:
```bash
python3 -m utils.make_data --name test --seed 1234 --num_samples 10000 --num_obs 5 20 --num_nodes 20 50 100 --max_length 2 3 4
python3 -m utils.make_data --name val --seed 4321 --num_samples 10000 --num_obs 5 20 --num_nodes 20 50 100 --max_length 2 3 4
```

To train a model use:

```bash
python3 train.py --model <model> --val_dataset data/nop/2depots/const/50/val_seed4321_T3_5-20obs.pkl --num_nodes 50 --max_length 3 --num_obs 5 20 --max_nodes 0 --combined_mha T --baseline critic --num_dirs 8
```

and change the environment conditions (number of nodes, max length, number of directions, validation dataset...) at your convenience. Available models are `naviformer` (NaviFormer), `pn` (Pointer Network), and `gpn` (Graph Pointer Network).

To resume training, run the same command to train and include the option `--resume path/to/model`.

Evaluate your trained models (in folder `outputs`) with:
```bash
python3 test.py <path/to/dataset> --model <path/to/model>
```
If the epoch is not specified, the last one in the folder will be used by default.

Route planning algorithms like OR-Tools (`ortools`) and Genetic Algorithm (`ga`) can be combined with path planners
like A-Star (`a_star`), D-Star (`d_star`), and Neural A-Star (`na_star`) as follows:
```bash
python3 -m benchmarks.nop.bench --route_planner <route_planner> --path_planner <path_planner> --datasets <path/to/dataset>
```

Finally, you can visualize an example using:
```bash
python3 visualize.py --model <path/to/model> --num_nodes 50 --max_length 3 --num_obs 5 20 --max_nodes 0
python3 visualize.py --model <route_planner>-<path_planner> --num_nodes 50 --max_length 3 --num_obs 5 20 --max_nodes 0
```
and change the model or scenario conditions at your convenience.

### Other options and help
```bash
python3 -m utils.make_data -h
python3 train.py -h
python3 test.py -h
python3 -m benchmarks.nop.bench -h
python3 visualize.py -h
```

## Acknowledgements
This implementation is based on the following repositories:
* [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)
* [AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
* [qiang-ma/graph-pointer-network](https://github.com/qiang-ma/graph-pointer-network)
* [omron-sinicx/neural-astar](https://github.com/omron-sinicx/neural-astar)
