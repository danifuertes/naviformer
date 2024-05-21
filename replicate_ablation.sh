# NaviFormer
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model outputs/nop_const50/naviformer --use_cuda false

# NaviFormer (2-step)
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model outputs/nop_const50/naviformer_2step --use_cuda false

# w/ standard Transformer encoder
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model outputs/nop_const50/naviformer_NormalEnc --use_cuda false

# w/ PN endoced-decoder
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model outputs/nop_const50/pn --use_cuda false

# w/ GPN encoder-decoder
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model outputs/nop_const50/gpn --use_cuda false

# w/ global maps
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model outputs/nop_const50/naviformer_GlobalMaps --use_cuda false

# w/o maps
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model outputs/nop_const50/naviformer_NoMaps --use_cuda false

# w/ 4 directions
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model outputs/nop_const50/naviformer_4dirs --use_cuda false
