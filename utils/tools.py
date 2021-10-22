import numpy as np
from pathlib import Path
import glob
import os
import yaml
import re
from utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string
# from icecream import ic


def load_config(cfg_path):
    with open(cfg_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)
    return file_cfg


def flops_info(model, input_shape=(3, 320, 320)):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False)
    flops = flops_to_string(flops * 2)
    params = params_to_string(params)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/*last.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file):
        return file
    # elif file == '' or file is None
    else: # file is fold
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found in Path: "%s"' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file


def create_workspace(cfg, resume=False):
    save_path = cfg['save_path']
    proj_name = cfg['proj_name']
    if resume:
        if 'weights_path' not in cfg:
            weights_dir = get_latest_run(save_path)
            assert weights_dir != '', \
                'last weights not exist in current resume path: {}'.format(os.path.abspath(save_path))
            log_dir = Path(weights_dir).parent.parent.as_posix()
        else:
            assert cfg.weights_path != '' and cfg.weights_path is not None, 'The Key "weights_path" is illegal in .yaml File'
            weights_dir = check_file(cfg['weights_path'])
            log_dir = increment_path(Path(save_path) / '{}_exp'.format(proj_name), exist_ok=False)
    else:
        # mkdir(rank, work_fold)
        log_dir = increment_path(Path(save_path) / '{}_exp'.format(proj_name), exist_ok=False)
        weights_dir = (Path(log_dir) / 'weights/last.pt').as_posix()
        (Path(log_dir) / 'weights').mkdir(parents=True)
    return log_dir, weights_dir