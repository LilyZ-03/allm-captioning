import torch
import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

VALID_DATASETS = ["small", "audiocaps", "clotho", "compa"]
VALID_EXPERIMENTS = ["generation", "selection", "matching"]
GENERATED_FILE_DIR = "/home/l1lyzhang/Generation/generated-data/"

def get_selection_prompt(captionA, captionB):
    return "CaptionA: %s CaptionB: %s Which caption more correctly describes the audio? Respond with CaptionA or CaptionB." % (captionA, captionB)

def get_matching_prompt(caption):
    # correctly describes ?
    return "Here is a caption that tries to describe the audio: %s Does the provided caption corresponds to the audio? Respond with Yes or No." % (caption)

def get_generation_prompt():
    generation_prompts = ["Describe the audio in details.", "Describe the audio in details with one sentence.", "Describe the audio."]
    return generation_prompts[2]

def get_captioning_data(mode):
    
    if mode == "small":
        json_file = "/home/l1lyzhang/Captioning-Datasets/test_clotho_v2.1.json"
        f = open(json_file, 'r')
        caps_data = json.load(f)
        f.close()
        return caps_data 
    elif mode == "audiocaps":
        json_file_list = ["test_audiocaps.json", "val_audiocaps.json"]
    elif mode == "clotho":
        json_file_list = ["test_clotho_v2.1.json", "val_clotho_v2.1.json"]
    
    # print(json_file_list)
    
    json_dir = "/home/l1lyzhang/Captioning-Datasets/"
    caps_data = []
    
    for file in json_file_list:
        f = open(json_dir + file)
        d = json.load(f)
        caps_data.extend(d)
        f.close()
    return caps_data


def get_compa_data():
    
    json_dir = "/home/l1lyzhang/CompA-dataset/CompA-json/"
    json_file_list = os.listdir(json_dir)
    # print(json_file_list)
    
    caps_data, original_data = [], []
    
    for file in json_file_list:
        f = open(json_dir + file)
        d = json.load(f)
        original_data.extend(d)
        f.close()
        
    for block in original_data:
        new_block = [{}, {}]
        new_block[0]["audio_path"] = block["audio_path"][0]
        new_block[1]["audio_path"] = block["audio_path"][1]
        caps_data.extend(new_block)
        
    return caps_data, original_data

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="small")
    parser.add_argument("--experiment", type=str, default="generation")
    args = parser.parse_args()
    
    # verify arguments
    if args.dataset not in VALID_DATASETS:
        raise ValueError("Invalid Dataset")
    elif args.experiment not in VALID_EXPERIMENTS:
        raise ValueError("Invalid Experiment")

    if args.dataset == "compa":
        caps_data, original_data = get_compa_data()
    else:
        caps_data = get_captioning_data(args.dataset)
        
    new_caps_data = json.loads(json.dumps(caps_data))
    
    if args.experiment == "generation":
        prompt = get_generation_prompt()

    # Set the following seed to reproduce our results.
    torch.manual_seed(1234)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("usage:", device)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
    model = model.to(device)

    for i in tqdm(range(len(caps_data))):
        
        block = caps_data[i] 
        audio_path = block["audio_path"]

        query = tokenizer.from_list_format([
            #{'audio': 'assets/audio/1272-128104-0000.flac'},
            {'audio': audio_path},
            # {'text': 'Who is speaking in the audio?'},
            {'text': prompt},
        ])

        response, history = model.chat(tokenizer, query=query, history=None)
        print(response)
        new_caps_data[i]["generated"] = response
        new_caps_data[i]["generation-prompt"] = prompt
        
    if args.dataset == "compa":
        output_data = original_data
        for i in range(len(output_data)):
            output_data[i]["generated"] = [new_caps_data[2 * i]["generated"], new_caps_data[2 * i + 1]["generated"]]
            output_data[i]["generation-prompt"] = prompt
    else:  
        output_data = new_caps_data
        
    generated_json = open(GENERATED_FILE_DIR + "qwen-audio-generated-captions-%s-2.json" % (args.dataset), 'w')
    json.dump(output_data, generated_json, indent=4)
    generated_json.close()