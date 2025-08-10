# ---------------------------------------------
# Evaluation code for E.T. Bench
# Copyright (c) 2024 Ye Liu
# Licensed under CC BY-NC-SA 4.0 license
# ---------------------------------------------
import sys
sys.path.append('.')
import argparse
import copy
import random
import re
import string

import nncore
import numpy as np
import torch
from nncore.ops import temporal_iou
from tabulate import tabulate

import sentence_transformers
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from sentence_transformers.util import dot_score

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from momentDETR_eval import vtg_eval

def get_boundaries(sample, th, nms=True, nms_th=0.7, use_tst=True):
    vl_sims = torch.tensor(sample["vl_sims"])
    if len(vl_sims) == 0:
        print("Didn't find any similarities")
        return [[0, sample["duration"], 1]], torch.ones(int(sample["duration"]))
    
    vl_sims_sm = (vl_sims * sample["scale"]).softmax(dim=1)

    ent_inds = torch.tensor(sample["ent_inds"])
    tst_inds = torch.tensor(sample["tst_inds"])
    inds = torch.cat([ent_inds, tst_inds], dim=0).sort().values

    ent_sims = [vl_sims[i] for i, idx in enumerate(inds) if idx in ent_inds]
    ent_sims_sm = [vl_sims_sm[i] for i, idx in enumerate(inds) if idx in ent_inds]
    if len(ent_sims):
        ent_sims = torch.stack(ent_sims)
        ent_sims_sm = torch.stack(ent_sims_sm)

    preds = []
    if len(ent_inds):
        for i, idx in enumerate(ent_inds):
            class_idx = torch.where(inds == idx)[0][0]
            class_score = ent_sims[i]
            class_score_sm = ent_sims_sm[i]

            rescaled = (class_score - class_score.amin()) / (class_score.amax() - class_score.amin())
            pred_binary = rescaled > 0.7
            
            if use_tst:
                other_score_sm = vl_sims_sm[inds != idx]
                if len(other_score_sm) > 0:
                    pred_binary = pred_binary & (class_score_sm >= other_score_sm.amax(dim=0))

                pred_binary_shifted_left = torch.roll(pred_binary, -1)
                pred_binary_shifted_left[-1] = 0
                pred_binary_shifted_right = torch.roll(pred_binary, 1)
                pred_binary_shifted_right[0] = 0
                pred_binary = pred_binary | pred_binary_shifted_left | pred_binary_shifted_right

            preds.append(pred_binary)

    boundaries = []
    confidence = []
    indices = []
    for i, pred in enumerate(preds):
        if len(ent_inds) == 0:
            continue
        ent_idx = ent_inds[i]
        class_idx = torch.where(inds == ent_idx)[0]

        boundary = torch.where(pred == 1)[0]
        if len(boundary) == 0:
            continue
        
        # merge the the indices that are close to each other
        boundary = boundary.cpu().numpy()
        boundary = np.split(boundary, np.where(np.diff(boundary) > 1)[0] + 1)
        boundary = [[b[0], b[-1]] if b[0] != b[-1] else [b[0], b[-1] + 1] for b in boundary] 
        conf = [vl_sims_sm[class_idx, b[0]:b[1] + 1] for b in boundary]
        boundaries.extend(boundary)
        confidence.extend(conf)
        indices.extend([i] * len(boundary))

    boundaries = torch.tensor(boundaries)
    if len(boundaries) == 0:
        print("Didn't find any boundaries")
        print(sample)
        return [[0, sample["duration"], 1]], torch.ones(int(sample["duration"])) 
    
    confidence_scalar = torch.tensor([c.max() for c in confidence])
    boundaries = torch.stack([boundaries[:, 0], boundaries[:, 1], confidence_scalar], dim=1)

    indices = torch.tensor(indices)
    ### NMS
    nms_cfg = dict(type='normal', thres=nms_th)
    for i in range(boundaries.size(0)):
        max_idx = boundaries[i:, -1].argmax(dim=0)
        boundaries = nncore.swap_element(boundaries, i, max_idx + i)
        indices = nncore.swap_element(indices, i, max_idx + i)
        iou = temporal_iou(boundaries[i, None, :-1], boundaries[i + 1:, :-1])[0]
        boundaries[i + 1:, -1][iou >= nms_cfg['thres']] = 0

    _, sort_idces = boundaries[:, -1].sort(descending=True)
    boundaries = boundaries[sort_idces]
    indices = indices[sort_idces].tolist()

    boundaries = boundaries.tolist()
    return boundaries, ent_sims_sm.mean(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path')
    parser.add_argument('--th', type=float, default=0.7)
    parser.add_argument('--nms_th', type=float, default=0.7)
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--task', type=str, default="all")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.pred_path.endswith('.json') or args.pred_path.endswith('.jsonl'):
        pred_paths = [args.pred_path]
        dir_name = nncore.dir_name(args.pred_path)
    else:
        pred_paths = nncore.ls(args.pred_path, ext=['json', 'jsonl'], join_path=True)
        pred_paths = [path for path in pred_paths if path != 'metrics.json']
        dir_name = args.pred_path

    log_file = nncore.join(dir_name, 'metrics.log')
    nncore.set_default_logger('etbench', fmt=None, log_file=log_file)

    nncore.log(f'Total number of files: {len(pred_paths)}')

    all_samples = []
    for path in pred_paths:
        nncore.log(f'Loading {path}...')
        all_samples += nncore.load(path)

    nncore.log(f'Total number of samples: {len(all_samples)}')

    pred = dict()
    for sample in all_samples:
        boundaries, scores = get_boundaries(sample, args.th, nms=True, nms_th=args.nms_th, use_tst="charades" in args.pred_path)
        scores = scores[:len(scores) - len(scores) % 2].reshape(-1, 2).mean(1).tolist()

        sample["pred_relevant_windows"] = boundaries
        sample["pred_saliency_scores"] = scores

    metrics = vtg_eval(all_samples, all_samples)
    for key, value in metrics["brief"].items():
        print(f"{key}: {value}")

    