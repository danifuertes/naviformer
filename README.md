# Route solver
![](images/python-3.8.svg)
![](images/torch-1.12.1.svg)
![](images/cuda-10.2.svg)
![](images/cudnn-7.6.svg)

![](images/top.gif)

This repository provides a framework to train, test, and validate neural networks for routing problems using deep
reinforcement learning.

## Dependencies

* Python == 3.10
* PyTorch == 1.12.1
* Cuda == 10.2
* Cudnn == 7.6
* Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

First, it is necessary to create test and validation sets:
```bash
python3 -m utils.make_data --name test --seed 1234 --num_samples 10000 --data_dist const --num_depots 2 --num_obs 5 20 --max_length 2 3 4
python3 -m utils.make_data --name val --seed 4321 --num_samples 10000 --data_dist const --num_depots 2 --num_obs 5 20 --max_length 2 3 4
```

To train the network (`naviformer`) use:
```bash
python3 train.py --model naviformer --val_dataset data/nop/2depots/const/50/val_seed4321_T3_5-20obs.pkl --num_nodes 50 --data_dist const --num_depots 2 --max_length 3 --num_obs 5 20 --max_nodes 0 --combined_mha T --baseline critic --num_dirs 8
```

and change the environment conditions (number of nodes, max length, reward distribution...) at your convenience.

Evaluate your trained models (in folder `outputs`) with:
```bash
python3 test.py data/nop/2depots/const/20/test_seed1234_L2_5obs.pkl --model outputs/np_const20/<model>
python3 test.py data/nop/2depots/const/50/test_seed1234_L3_5obs.pkl --model outputs/nop_const50/<model>
python3 test.py data/nop/2depots/const/100/test_seed1234_L4_5obs.pkl --model outputs/nop_const100/<model>
```
If the epoch is not specified, the last one in the folder will be used by default.

Route planning algorithms like OR-Tools (`ortools`) and Genetic Algorithm (`ga`) can be combined with path planners
like A-Star (`a_star`) and D-Star (`d_star`) as follows:
```bash
python3 -m benchmarks.nop.bench --route_planner ortools --path_planner a_star --datasets data/nop/2depots/const/20/test_seed1234_L2_5obs.pkl --multiprocessing T
python3 -m benchmarks.nop.bench --route_planner ortools --path_planner d_star --datasets data/nop/2depots/const/20/test_seed1234_L2_5obs.pkl --multiprocessing T
python3 -m benchmarks.nop.bench --route_planner ga --path_planner a_star --datasets data/nop/2depots/const/20/test_seed1234_L2_5obs.pkl --multiprocessing T
python3 -m benchmarks.nop.bench --route_planner ga --path_planner d_star --datasets data/nop/2depots/const/20/test_seed1234_L2_5obs.pkl --multiprocessing T
```

Finally, you can visualize an example using:
```bash
python3 visualize.py --model outputs/nop_const20/<model> --num_depots 2 --num_nodes 20 --max_length 3 --data_dist const --max_obs 5 --max_nodes 0
python3 visualize.py --model ortools-a_star --num_depots 2 --num_nodes 20 --max_length 3 --data_dist const --max_obs 5 --max_nodes 0
```

### Other options and help
```bash
python3 -m utils.make_data -h
python3 train.py -h
python3 test.py -h
python3 -m benchmarks.nop.bench -h
python3 visualize.py -h
```

### Coming soon...
* Implement your own network
* Implement your own benchmark algorithm
* Implement your own environment
* RoboMaster demo

## Acknowledgements
This repository is an adaptation of
[wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) for the TOP. The baseline
algorithms (A-Star, D-Star) were implemented following
[AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics).