from easydict import EasyDict as edict
import json

## test set location
config = edict()
config.TEST = edict()
config.TEST.folder_path_c = './DPD/test_c/source/' # need to change
config.TEST.folder_path_gt = './DPD/test_c/target/' # need to change

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
