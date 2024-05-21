# NaviFormer
python test.py data/PASTIS/data_max100.pkl --model outputs/nop_const100max/naviformer --batch_size 1 --use_cuda false

# NaviFormer + NA*
python test.py data/PASTIS/data_max100.pkl --model outputs/op_const100max/naviformer_na_star --batch_size 1 --use_cuda false

# NaviFormer + A*
python test.py data/PASTIS/data_max100.pkl --model outputs/op_const100max/naviformer_a_star --batch_size 1 --use_cuda false

# # PN + CNN
# python test.py data/PASTIS/data.pkl --model outputs/nop_const50/pn --batch_size 1 --use_cuda true

# # PN + NA*
# python test.py data/PASTIS/data_max100.pkl --model outputs/op_const100max/pn_na_star --batch_size 1 --use_cuda true

# # PN + A*
# python test.py data/PASTIS/data_max100.pkl --model outputs/op_const100max/pn_a_star --batch_size 1 --use_cuda true

# # GPN + CNN
# python test.py data/PASTIS/data.pkl --model outputs/nop_const50/gpn --batch_size 1 --use_cuda true

# # GPN + NA*
# python test.py data/PASTIS/data_max100.pkl --model outputs/op_const100max/gpn_na_star --batch_size 1 --use_cuda true

# # GPN + A*
# python test.py data/PASTIS/data_max100.pkl --model outputs/op_const100max/gpn_a_star --batch_size 1 --use_cuda true

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
