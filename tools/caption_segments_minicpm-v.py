import os
import re
import sys
import argparse
import json
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer
from einops import rearrange
from torch.multiprocessing import Process, Queue, Value, Lock, set_start_method
import time
import tqdm

torch.manual_seed(0)

# define argparser
parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, required=True)
parser.add_argument('--ann_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--num_gpus', type=int, default=4)
parser.add_argument('--num_frames', type=int, default=64)
parser.add_argument('--task', type=str)
parser.add_argument('--v100', type=bool)
parser.add_argument('--attention', type=str, default="flash_attention_2")
args = parser.parse_args()

MODEL_NAME = "openbmb/MiniCPM-V-2_6" # or local model path
MAX_NUM_FRAMES = args.num_frames


if args.v100:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    
def encode_video(video_path, clips=None):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    frames = []
    if clips is not None:
        for clip in clips:
            clip_idx = frame_idx[clip[0]:clip[1]]

            if len(clip_idx) > MAX_NUM_FRAMES:
                clip_idx = uniform_sample(clip_idx, MAX_NUM_FRAMES)

            frames_ = vr.get_batch(clip_idx).asnumpy()
            frames_ = [Image.fromarray(v.astype('uint8')) for v in frames_]
            frames.append(frames_) 
    else: 
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        frames = [frames]
    return frames

def prompt_vlm(model, tokenizer, frames, question):
    msgs = [
        {'role': 'user', 'content': frames + [question]}, 
    ]

    # Set decode params for video
    params = {}
    params["use_image_id"] = False
    params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution > 448*448

    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        **params
    )
    return answer

def process_videos(annotations, output_queue, video_dir, output_dir, device, attention, progress, lock):
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True,
    attn_implementation=attention, torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True) 
    
    new_annotations = []
    for idx in range(len(annotations)):
        try:
            ann = annotations[idx]
            video_path = os.path.join(video_dir, ann["video"])

            save_path = os.path.join(output_dir, f"{ann['id']}.json")
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    new_ann = json.load(f)
                new_annotations.append(new_ann)
                # Update progress
                with lock:
                    progress.value += 1
                continue

            if ann["task"] == "gvq":
                question = f"Based on the clip, find important visual details that helps find the answer '{ann['extracted_answer']}' to the question '{ann['extracted_query']}'."
            elif ann["task"] == "epm":
                question = f"Based on the clip, find important visual details related to the human activity '{ann['extracted_query']}'."
            elif ann["task"] == "tvg":
                question = f"Based on the clip, find important visual details related to the event '{ann['extracted_query']}'."
            elif ann["task"] == "tal":
                question = f"Based on the clip, find important visual details related to the action '{ann['extracted_query']}'."
            elif ann["task"] == "vhd":
                question = f"Based on the clip, find important visual details related to the event '{ann['extracted_query']}'."
            elif ann["task"] == "evs":
                question = f"Based on the clip, find important visual details related to the topic '{ann['extracted_query']}'."
            elif ann["task"] in ["dvc", "slc", "tvc"]:
                question_template = "Based on the clip, add more important visual details to the clip summary '{clip_summary}'"
                if ann["task"] in ["dvc", "slc"]:
                    # Regular expression to remove the time prefixes
                    pattern = r'\d+\.\d+\s*-\s*\d+\.\d+\s*seconds,\s*(.*?)(?=\d+\.\d+\s*-\s*\d+\.\d+\s*seconds,|$)'
                else: 
                    pattern = r'\d+\.\d+\s*seconds,\s*(.*?)(?=\d+\.\d+\s*seconds,|$)'
                # Regular expression to remove the time prefixes
                summaries = re.findall(pattern, ann["conversations"][1]["value"])
                # Split the cleaned text into sentences
                summaries = [sentence.strip().capitalize() for sentence in summaries]
            else: 
                raise NotImplementedError(f"Task {ann['task']} not implemented")

            if ann["task"] in ["tvc", "dvc", "slc"]:
                question_template += " Avoid subjective interpretation and focus on factual visual content. Be concise with your response."
            else:
                question += " Avoid subjective interpretation and focus on factual visual content. Be concise with your response."

            if ann["task"] in ["tvg", "tal", "evs",  "epm", "dvc", "slc"]: 
                # Window type: [start, end, start, end, ...]
                clips = ann["tgt"]
            elif ann["task"] == "gvq":
                clips = [max(ann["tgt"][0] - 3, 0), ann["tgt"][1] + 3]
            elif ann["task"] == "vhd":
                # Window type: [time]
                clips = [max(ann["tgt"][0] - 3, 0), ann["tgt"][0] + 3]
            elif ann["task"] == "tvc":
                clips = [(ann["tgt"][i], ann["tgt"][i+1]) for i in range(0, len(ann["tgt"]) - 1)]
                clips.append([ann["tgt"][-1], ann["duration"]])
            
            clips = np.array(clips).reshape(-1, 2).astype(int)
            video = encode_video(video_path, clips=clips)
            captions = []
            for j, frames in enumerate(video):
                if ann["task"] in ["tvc", "dvc", "slc"]:
                    question = question_template.format(clip_summary=summaries[j])
                answer = prompt_vlm(model, tokenizer, frames, question)
                
                # if j == 0:
                #     print(answer)

                captions.append(answer)
            new_annotations.append(ann)
            new_annotations[-1]["clips"] = clips.tolist() 
            new_annotations[-1]["captions"] = captions

            with open(save_path, 'w') as f:
                json.dump(new_annotations[-1], f, indent=4)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            with open(save_path, 'w') as f:
                json.dump(ann, f, indent=4)
            continue
        # Update progress
        with lock:
            progress.value += 1

    output_queue.put(new_annotations)

def distribute_work(annotations, task, video_dir, output_dir, num_gpus, attention):
    # Split video paths into chunks for each process (GPU)
    annotation_chunks = np.array_split(annotations, num_gpus)

    # Queue to store the results
    output_queue = Queue()

    # Shared progress value and lock for synchronization
    progress = Value('i', 0)  # 'i' indicates an integer
    total_files = len(annotations)
    lock = Lock()

    # Create processes
    processes = []
    for i in range(num_gpus):
        device = f'cuda:{i}'
        process = Process(target=process_videos, args=(annotation_chunks[i], output_queue, video_dir, output_dir, device, attention, progress, lock))
        processes.append(process)
    
    # Start processes
    for process in processes:
        process.start()

    # Progress monitoring loop
    with tqdm.tqdm(total=total_files, desc="Processing Videos", unit="video") as pbar:
        while progress.value < total_files:
            time.sleep(1)  # Sleep for a second between updates
            with lock:
                pbar.n = progress.value
                pbar.refresh()

    # Collect results
    results = []
    for _ in range(num_gpus):
        result = output_queue.get()
        results.extend(result)
    
    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Save the final result
    with open(os.path.join(output_dir, f'captions_{task}.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    # Set the start method to 'spawn' (better compatibility for CUDA)
    set_start_method('spawn', force=True)

    with open(args.ann_path, "r") as f:
        annotations = json.load(f)

    annotations = [ann for ann in annotations if ann['task'] == args.task]

    for i, ann in enumerate(annotations):
        ann['id'] = i 
        
    os.makedirs(args.output_dir, exist_ok=True)
    print("Processing", len(annotations), "videos")
    distribute_work(annotations, args.task, args.video_dir, args.output_dir, args.num_gpus, args.attention)
