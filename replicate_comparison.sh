# NaviFormer
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/naviformer --batch_size 1 --use_cuda false

# NaviFormer + NA*
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/naviformer_na_star --batch_size 1 --use_cuda false

# NaviFormer + A*
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/naviformer_a_star --batch_size 1 --use_cuda false

# NaviFormer + TransPath
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/naviformer_transpath --batch_size 1 --use_cuda false

# PN + CNN
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/pn --batch_size 1 --use_cuda false

# PN + NA*
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/pn_na_star --batch_size 1 --use_cuda false

# PN + A*
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/pn_a_star --batch_size 1 --use_cuda false

# PN + TransPath
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/pn_transpath --batch_size 1 --use_cuda false

# GPN + CNN
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/gpn --batch_size 1 --use_cuda false

# GPN + NA*
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/gpn_na_star --batch_size 1 --use_cuda false

# GPN + A*
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/gpn_a_star --batch_size 1 --use_cuda false

# GPN + TransPath
python test.py data/PASTIS/data_max100.pkl --model pretrained/comparison/gpn_transpath --batch_size 1 --use_cuda false

# OR-Tools + NA*
python -m benchmarks.nop.bench --datasets data/PASTIS/data.pkl --route_planner ortools --path_planner na_star --use_cuda false --eps 0.3 -f

# OR-Tools + A*
python -m benchmarks.nop.bench --datasets data/PASTIS/data.pkl --route_planner ortools --path_planner a_star --use_cuda false --eps 0.3 -f

# OR-Tools + D*
python -m benchmarks.nop.bench --datasets data/PASTIS/data.pkl --route_planner ortools --path_planner d_star --use_cuda false --eps 0.1 -f

# GA + NA*
python -m benchmarks.nop.bench --datasets data/PASTIS/data.pkl --route_planner ga --path_planner na_star --use_cuda false --eps 0.3 -f

# GA + A*
python -m benchmarks.nop.bench --datasets data/PASTIS/data.pkl --route_planner ga --path_planner a_star --use_cuda false --eps 0.3 -f

# GA + D*
python -m benchmarks.nop.bench --datasets data/PASTIS/data.pkl --route_planner ga --path_planner d_star --use_cuda false --eps 0.1 -f
