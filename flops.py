import torch
from model import build_model
# from model.arch import build_model
from utils import load_config, get_model_complexity_info


def main(config, input_shape=(3, 320, 320)):
    model = build_model(config.model, input_shape[: 1])
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')


def flops_info(model, input_shape=(3, 320, 320)):
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')



