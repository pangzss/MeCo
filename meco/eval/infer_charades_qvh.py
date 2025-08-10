import argparse
import copy
import nncore
import torch
import multiprocessing
from tqdm import tqdm
from multiprocessing import Value, Lock
from meco.constants import DEFAULT_IMAGE_TOKEN
from meco.conversation import get_conv
from meco.model.builder import build_model
from meco.utils.inference import KeywordsStoppingCriteria, RepetitionPenaltyLogitsProcessor
from meco.utils.io import load_video
from meco.utils.tokenization import detokenize, tokenize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_path')
    parser.add_argument('--data_path')
    parser.add_argument('--pred_path')
    parser.add_argument('--model_path')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dtype', type=str, default="fp16")
    return parser.parse_args()

def process_chunk(gpu_id, chunk_idx, args, anno_chunks, progress_counter, lock):
    device = f'cuda:{gpu_id}'
    pred_path = nncore.join(args.pred_path, f'{chunk_idx}.json')
    anno = anno_chunks[chunk_idx]

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model, tokenizer, transform = build_model(args.model_path, device=device, dtype=dtype)

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    for i in (range(len(anno))):
        sample = copy.deepcopy(anno[i])

        video = nncore.join(args.data_path, sample['video'])
        video, tag = load_video(video, num_threads=1)
        video = transform(video).to(dtype).to(device)

        query, src = sample['q'], sample.get('src')

        conv = get_conv(model.config.conv_type)
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenize(prompt, tokenizer).unsqueeze(0).to(device)
        stop_str = conv.seps[-1]
        stopping_criteria = [KeywordsStoppingCriteria(tokenizer, stop_str)]

        with torch.inference_mode():
            out = model.generate(
                input_ids,
                image=[video],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=500,
                stopping_criteria=stopping_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True,
                query=[[query]],
                src=[src],
                tag=[tag]
            )

        tokens = out[0][0, input_ids.size(1):]
        response = tokenizer.decode(tokens, skip_special_tokens=False).strip()

        if response.endswith(stop_str):
            response = response[:-len(stop_str)].strip()

        print(response)
        anno[i]['a'] = response
        
        # gather similarities
        if not torch.is_tensor(tokens):
            tokens = torch.LongTensor(tokens)

        # ent token pred
        vl_sims = []
         
        ent_inds = torch.where(tokens == model.config.ent_token_id)[0]
            
        tst_inds = torch.where(tokens == model.config.tst_token_id)[0]
        
        inds = torch.cat([ent_inds, tst_inds], dim=0).sort().values
            
        for k in range(len(inds)):
            sim = model.sim[inds[k]]
            vl_sims.append(sim.tolist())

        anno[i]["ent_inds"] = ent_inds.tolist()
        anno[i]["tst_inds"] = tst_inds.tolist()

        anno[i]['vl_sims'] = vl_sims
        with lock:
            progress_counter.value += 1

    nncore.dump(anno, pred_path)

def update_progress(progress_counter, total_samples, lock):
    """Update the shared progress bar."""
    with tqdm(total=total_samples, desc="Processing") as pbar:
        while True:
            with lock:
                pbar.n = progress_counter.value
            pbar.refresh()
            if progress_counter.value >= total_samples:
                break
            
def main():
    args = parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected. Please ensure CUDA devices are available.")

    gpu_ids = list(range(num_gpus))

    anno = nncore.load(args.anno_path)
   
    for a in anno:
        if "vhd" in args.pred_path:
            query = f"You are given a video about daily activities. Watch the video carefully and find the highlight moment described by the sentence: '{a['query']}'. Provide a detailed description of the moment."
        else:
            query = f"You are given a video about daily activities. Watch the video carefully and find the visual event described by the sentence: '{a['query']}'. Provide a detailed description of the event."

        a['q'] = query
        a['video'] = a['vid'] + ".mp4"

    anno_chunks = [anno[i::num_gpus] for i in range(num_gpus)]

    # Use 'spawn' start method
    multiprocessing.set_start_method('spawn', force=True)

    # Shared progress counter and lock
    progress_counter = Value('i', 0)
    lock = Lock()

    # Start progress monitor
    progress_monitor = multiprocessing.Process(target=update_progress, args=(progress_counter, len(anno), lock))
    progress_monitor.start()

    # Start worker processes
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = multiprocessing.Process(target=process_chunk, args=(gpu_id, i, args, anno_chunks, progress_counter, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Stop progress monitor
    progress_monitor.join()

if __name__ == '__main__':
    main()