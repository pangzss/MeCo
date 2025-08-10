import os
import sys
sys.path.append('..')
import re
import random
import copy
import nncore
import torch
import json
import imageio
import cv2
import numpy as np
import argparse
import torch.backends.cudnn as cudnn
from meco.constants import DEFAULT_IMAGE_TOKEN
from meco.conversation import get_conv
from meco.model.builder import build_model
from meco.utils.inference import KeywordsStoppingCriteria, RepetitionPenaltyLogitsProcessor
from meco.utils.io import load_video
from meco.utils.tokenization import detokenize, tokenize
from transformers import AutoConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import Video
from nncore.ops import temporal_iou
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--video_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dtype', type=str, default="fp16")
    return parser.parse_args()


class MME_dataset(Dataset):
    def __init__(self, data_prefix, anno_path, transform, max_num_frames=128, max_subtitle_len=4096):
        self.data_prefix = data_prefix
        self.data_list = nncore.load(anno_path)

        self.max_num_frames = max_num_frames
        self.max_subtitle_len = max_subtitle_len
        
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def read_video(self, video_path, bound=None):
        video_raw, _ = load_video(video_path, max_len=self.max_num_frames, num_threads=1)
        video = self.transform(video_raw)
        return video

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer = f"({answer}) {data['options'][ord(answer) - ord('A')][3:]}"
        for idx, c in enumerate(data['options']):
            cur_choice, cur_text = c[0], c[3:]
            question += f"({cur_choice}) {cur_text}\n"
        question = question.rstrip()
        return question, answer

    def __getitem__(self, idx):
        video_name = self.data_list[idx]['videoID']
        video_path = os.path.join(self.data_prefix, "data", video_name + '.mp4')

        torch_imgs = self.read_video(video_path)

        duration_category = self.data_list[idx]['duration']
        qa_list = []
        qa_list.append(self.qa_template(self.data_list[idx]))

        subtitle = ""
                
        return {
            'subtitle': subtitle,
            'video': torch_imgs, 
            'qa_list': qa_list,
            'duration_category': duration_category,
        }

def infer_mme(
        model,
        data_sample, 
        tokenizer,
        system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        add_subtitle=False,
        n_frms=128
    ):

    video = data_sample["video"].to(model.dtype).to(model.device)

    pred_list = []
    gt_list = []
    for idx, qa in enumerate(data_sample['qa_list']):
        # print(f"----------qa_{idx}---------")
        query = system + qa[0] + question_prompt
        print(query)
        conv = get_conv(model.config.conv_type)
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
            
        input_ids = tokenize(prompt, tokenizer).unsqueeze(0).to(model.device)

        stop_str = conv.seps[-1]
        stopping_criteria = [KeywordsStoppingCriteria(tokenizer, stop_str)]

        with torch.inference_mode():
            out = model.generate(
                input_ids,
                image=[video],
                do_sample=False,
                max_new_tokens=500,#2048,
                cache_implementation=None,
                stopping_criteria=stopping_criteria,
                tag=[None],
                src=[None],
                query=[[query]],
                output_hidden_states=True,
                return_dict_in_generate=True)

        tokens = out[0][0, input_ids.size(1):]
        # response = tokenizer.decode(tokens, skip_special_tokens=False).strip()
        response = detokenize(tokens, model, tokenizer)
        if response.endswith(stop_str):
            response = response[:-len(stop_str)].strip()
        # remove potential explanation
        if response[:6] == "Answer":
            response = response.split("Answer: ")[-1]
        pred = response.replace("(", "").replace(")", "")[0]

        pred_list.append(pred.lower())
        gt_list.append(qa[1][1].lower())
        print(f"Pred: {pred}")
        print(f"GT: {qa[1][1]}")
    return pred_list, gt_list


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # load model
    device = f"cuda:{args.gpu}"
    model_path = args.model_path
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model, tokenizer, transform = build_model(model_path, device=device, dtype=dtype)

    dataset = MME_dataset(args.data_dir, args.anno_path, transform, max_num_frames=args.max_num_frames)

    res_json_data = nncore.load(args.anno_path)

    correct = 0
    total = 0
    res_list = []
    acc_dict = {}

    output_dir = os.path.join(model_path, "videomme")

    for idx, example in enumerate(tqdm(dataset)):
        
        duration_category = example['duration_category']
        if duration_category not in acc_dict:
            acc_dict[duration_category] = [0, 0] # correct, total
        qa_count = len(example['qa_list'])
        acc_dict[duration_category][1] += qa_count
        total += qa_count
        pred_list, gt_list = infer_mme(
            model,
            example, 
            tokenizer,
            "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n",
            question_prompt="\nOnly give the best option.",
            answer_prompt="Best option:(",
            return_prompt='(',
            system_q=False,
            print_res=False,
            system_llm=True,
            # add_subtitle=True, # Comment this line to add subtitles, we use the whole subtitles by default.
        )
        res_list.append({
            'pred': pred_list,
            'gt': gt_list
        })
        qa_idx = 0
        for pred, gt in zip(pred_list, gt_list):
            if pred == gt:
                acc_dict[duration_category][0] += 1
                correct += 1
            res_json_data[idx]['response'] = pred
            qa_idx += 1
        print(f"Part  Acc: {acc_dict[duration_category][0] / acc_dict[duration_category][1] * 100 :.2f}%")
        print(f"Total Acc: {correct / total * 100 :.2f}%")
        print('-' * 50, duration_category, '-' * 50)

    with open(f"{output_dir}/test.json", "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

    final_res = dict()
    correct = 0
    total = 0
    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]
    final_res['Avg'] = correct / total * 100

    print(final_res)
    
    with open(f"{output_dir}/upload_leaderboard.json", "w+") as f:
        json.dump(final_res, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dtype', type=str, default="fp16")
    parser.add_argument('--anno_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--max_num_frames', type=int, default=128)
    args = parser.parse_args()
    main(args)