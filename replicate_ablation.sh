# NaviFormer
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model pretrained/ablation/naviformer --use_cuda false

# NaviFormer (2-step)
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model pretrained/ablation/naviformer_2step --use_cuda false

# w/ standard Transformer encoder
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model pretrained/ablation/naviformer_NormalEnc --use_cuda false

# w/ PN endoced-decoder
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model pretrained/ablation/pn --use_cuda false

# w/ GPN encoder-decoder
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model pretrained/ablation/gpn --use_cuda false

# w/ global maps
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model pretrained/ablation/naviformer_GlobalMaps --use_cuda false

# w/o maps
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model pretrained/ablation/naviformer_NoMaps --use_cuda false

# w/ 4 directions
python test.py data/nop/2depots/const/50/test_seed1234_T3_5-20obs.pkl --model pretrained/ablation/naviformer_4dirs --use_cuda false
