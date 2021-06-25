# Copyright 2021 Sea Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted for VOLO
'''
- load_pretrained_weights: load pretrained paramters to model in transfer learning
- resize_pos_embed: resize position embedding
- get_mean_and_std: calculate the mean and std value of dataset.
'''
import torch
import math

import logging
import os
from collections import OrderedDict
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


def resize_pos_embed(posemb, posemb_new):
    '''
    resize position embedding with class token
    example: 224:(14x14+1)-> 384: (24x24+1)
    return: new position embedding
    '''
    ntok_new = posemb_new.shape[1]

    posemb_tok, posemb_grid = posemb[:, :1], posemb[0,1:]  # posemb_tok is for cls token, posemb_grid for the following tokens
    ntok_new -= 1
    gs_old = int(math.sqrt(len(posemb_grid)))  # 14
    gs_new = int(math.sqrt(ntok_new))  # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(
        0, 3, 1, 2)  # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(
        posemb_grid, size=(gs_new, gs_new),
        mode='bicubic')  # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, gs_new * gs_new, -1)  # [1, dim, 24, 24] -> [1, 24*24, dim]
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)  # [1, 24*24+1, dim]
    return posemb


def resize_pos_embed_without_cls(posemb, posemb_new):
    '''
    resize position embedding without class token
    example: 224:(14x14)-> 384: (24x24)
    return new position embedding
    '''
    ntok_new = posemb_new.shape[1]
    posemb_grid = posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))  # 14
    gs_new = int(math.sqrt(ntok_new))  # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(
        0, 3, 1, 2)  # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(
        posemb_grid, size=(gs_new, gs_new),
        mode='bicubic')  # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, gs_new * gs_new, -1)  # [1, dim, 24, 24] -> [1, 24*24, dim]
    return posemb_grid


def resize_pos_embed_4d(posemb, posemb_new):
    '''return new position embedding'''
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    gs_old = posemb.shape[1]  # 14
    gs_new = posemb_new.shape[1]  # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb
    posemb_grid = posemb_grid.permute(0, 3, 1,
                                      2)  # [1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic')  # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1)  # [1, dim, 24, 24]->[1, 24, 24, dim]
    return posemb_grid

def load_state_dict(checkpoint_path, model, use_ema=False, num_classes=1000):
    # load state_dict
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(
            state_dict_key, checkpoint_path))
        if num_classes != 1000:
            # completely discard fully connected for all other differences between pretrained and created model
            del state_dict['head' + '.weight']
            del state_dict['head' + '.bias']
            old_aux_head_weight = state_dict.pop('aux_head.weight', None)
            old_aux_head_bias = state_dict.pop('aux_head.bias', None)

        old_posemb = state_dict['pos_embed']
        if model.pos_embed.shape != old_posemb.shape:  # need resize the position embedding by interpolate
            if len(old_posemb.shape) == 3:
                if int(math.sqrt(
                        old_posemb.shape[1]))**2 == old_posemb.shape[1]:
                    new_posemb = resize_pos_embed_without_cls(
                        old_posemb, model.pos_embed)
                else:
                    new_posemb = resize_pos_embed(old_posemb, model.pos_embed)
            elif len(old_posemb.shape) == 4:
                new_posemb = resize_pos_embed_4d(old_posemb, model.pos_embed)
            state_dict['pos_embed'] = new_posemb

        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained_weights(model,
                            checkpoint_path,
                            use_ema=False,
                            strict=True,
                            num_classes=1000):
    '''load pretrained weight for VOLO models'''
    state_dict = load_state_dict(checkpoint_path, model, use_ema, num_classes)
    model.load_state_dict(state_dict, strict=strict)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
