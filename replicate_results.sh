# 20 nodes
python test.py data/nop/2depots/const/20/scenario_seed0_0-20obs.pkl data/nop/2depots/const/20/scenario_seed0_5-20obs.pkl data/nop/2depots/const/20/scenario_seed0_10-20obs.pkl data/nop/2depots/const/20/scenario_seed0_15-20obs.pkl data/nop/2depots/const/20/scenario_seed0_20-20obs.pkl --model outputs/nop_const100max/naviformer --use_cuda false

# 50 nodes
python test.py data/nop/2depots/const/50/scenario_seed0_0-20obs.pkl data/nop/2depots/const/50/scenario_seed0_5-20obs.pkl data/nop/2depots/const/50/scenario_seed0_10-20obs.pkl data/nop/2depots/const/50/scenario_seed0_15-20obs.pkl data/nop/2depots/const/50/scenario_seed0_20-20obs.pkl --model outputs/nop_const100max/naviformer --use_cuda false

# 100 nodes
python test.py data/nop/2depots/const/100/scenario_seed0_0-20obs.pkl data/nop/2depots/const/100/scenario_seed0_5-20obs.pkl data/nop/2depots/const/100/scenario_seed0_10-20obs.pkl data/nop/2depots/const/100/scenario_seed0_15-20obs.pkl data/nop/2depots/const/100/scenario_seed0_20-20obs.pkl --model outputs/nop_const100max/naviformer --use_cuda false
