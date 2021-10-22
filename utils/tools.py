import numpy as np
from pathlib import Path
import glob
import os
import re
import torch
import torch.nn.functional as F
from utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string
# from icecream import ic




class SSDBox(object):
    def __init__(self, is_normalized=True):
        self.is_normalized = is_normalized
        self.norm_delta = float(not self.is_normalized)

    def __call__(self,
                 preds,
                 prior_boxes,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes, scores = preds
        outputs = []
        # ic(prior_boxes.shape)
        # ic(scores.shape)
        #
        # ic(boxes.shape)
        prior_box = prior_boxes
        for box, score in zip(boxes, scores):
            # print(prior_box.shape)
            pb_w = prior_box[:, 2] - prior_box[:, 0] + self.norm_delta
            pb_h = prior_box[:, 3] - prior_box[:, 1] + self.norm_delta
            pb_x = prior_box[:, 0] + pb_w * 0.5
            pb_y = prior_box[:, 1] + pb_h * 0.5
            out_x = pb_x + box[:, :, 0] * pb_w * 0.1
            out_y = pb_y + box[:, :, 1] * pb_h * 0.1
            out_w = torch.exp(box[:, :, 2] * 0.2) * pb_w
            out_h = torch.exp(box[:, :, 3] * 0.2) * pb_h

            if self.is_normalized:
                h = torch.unsqueeze(
                    im_shape[:, 0] / scale_factor[:, 0], dim=-1)
                w = torch.unsqueeze(
                    im_shape[:, 1] / scale_factor[:, 1], dim=-1)
                output = torch.stack([(out_x - out_w / 2.) * w,
                                      (out_y - out_h / 2.) * h,
                                      (out_x + out_w / 2.) * w,
                                      (out_y + out_h / 2.) * h], dim=-1)
            else:
                output = torch.stack([out_x - out_w / 2., out_y - out_h / 2., out_x + out_w / 2. - 1., out_y + out_h / 2. - 1.], dim=-1)
            outputs.append(output)
        boxes = torch.cat(outputs, dim=1)

        scores = F.softmax(torch.cat(scores, dim=-1))
        scores = torch.transpose(scores, 1, 2)

        return boxes, scores


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
    if resume:
        if 'weights_path' not in cfg:
            weights_dir = get_latest_run(cfg.save_path)
            assert weights_dir != '', \
                'last weights not exist in current resume path: {}'.format(os.path.abspath(cfg.save_path))
            log_dir = Path(weights_dir).parent.parent.as_posix()
        else:
            assert cfg.weights_path != '' and cfg.weights_path is not None, 'The Key "weights_path" is illegal in .yaml File'
            weights_dir = check_file(cfg.weights_path)
            log_dir = increment_path(Path(cfg.save_path) / '{}_exp'.format(cfg.proj_name), exist_ok=False)
    else:
        # mkdir(rank, work_fold)
        log_dir = increment_path(Path(cfg.save_path) / '{}_exp'.format(cfg.proj_name), exist_ok=False)
        weights_dir = (Path(log_dir) / 'weights/last.pt').as_posix()
        (Path(log_dir) / 'weights').mkdir(parents=True)
    return log_dir, weights_dir