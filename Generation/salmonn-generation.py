import torch
import argparse
import os
import json
from tqdm import tqdm
from model import SALMONN

VALID_DATASETS = ["small", "audiocaps", "clotho", "compa"]
VALID_EXPERIMENTS = ["generation", "selection", "matching"]

def get_selection_prompt(captionA, captionB):
    return "CaptionA: %s CaptionB: %s Which caption more correctly describes the audio? Respond with CaptionA or CaptionB." % (captionA, captionB)

def get_matching_prompt(caption):
    # correctly describes ?
    return "Here is a caption that tries to describe the audio: %s Does the provided caption corresponds to the audio? Respond with Yes or No." % (caption)

def get_generation_prompt():
    generation_prompts = ["Describe the audio in details.", "Describe the audio in details with one sentence.", "Describe the audio."]
    return generation_prompts[2]

def get_captioning_data():
    json_dir = "/home/l1lyzhang/Captioning-Datasets/"
    # json_file_list = os.listdir(json_dir)
    json_file_list = ["test_audiocaps.json", "val_audiocaps.json"]
    print(json_file_list)
    
    caps_data = []
    for file in json_file_list:
        f = open(json_dir + file)
        d = json.load(f)
        caps_data.extend(d)
    
    # json_file = "/home/l1lyzhang/Captioning-Datasets/test_clotho_v2.1.json"
    # f = open(json_file, 'r')
    # caps_data = json.load(f)
    # f.close()

    return caps_data 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt_path", type=str, default="/home/l1lyzhang/salmonn_v1.pth")
    parser.add_argument("--whisper_path", type=str, default="/home/l1lyzhang/SALMONN/whisper-large-v2")
    parser.add_argument("--beats_path", type=str, default="/home/l1lyzhang/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    parser.add_argument("--vicuna_path", type=str, default="/home/l1lyzhang/SALMONN/vicuna-13b-v1.1")
    parser.add_argument("--low_resource", action='store_true', default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    
    caps_data = get_captioning_data()
    new_caps_data = json.loads(json.dumps(caps_data))

    model = SALMONN(
        ckpt=args.ckpt_path,
        whisper_path=args.whisper_path,
        beats_path=args.beats_path,
        vicuna_path=args.vicuna_path,
        low_resource=args.low_resource
    )
    model.to(args.device)
    model.eval()
    
    prompt = "Describe the audio in details with one sentence."
    
    for i in tqdm(range(len(caps_data))):
        block = caps_data[i]
        wav_path = block["audio_path"]

        try:
            # for environment with cuda>=117
            with torch.cuda.amp.autocast(dtype=torch.float16):
                response = model.generate(wav_path, prompt=prompt)[0]
                print(response)
                new_caps_data[i]["generated"] = response
                new_caps_data[i]["generation-prompt"] = prompt
        except Exception as e:
            print(e)
            if args.debug:
                import pdb; pdb.set_trace()

    generated_json = open("/home/l1lyzhang/salmonn-generated-captions-small-2.json", 'w')
    json.dump(new_caps_data, generated_json, indent=4)
    generated_json.close()