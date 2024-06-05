import numpy as np
import json
import os
# from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def get_cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1,-1)
    embedding2 = np.array(embedding2).reshape(1,-1)
    
    return cosine_similarity(embedding1, embedding2)[0][0]

def get_embedding_ada(client, text):
    return client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    ).data[0].embedding

def get_embedding_similarity_ada(client, ground_truth_text, generated_text):
    
    generated_embedding = get_embedding_ada(client, generated_text)
    similarity_by_caption = np.array([get_cosine_similarity(get_embedding_ada(client, i), generated_embedding) for i in ground_truth_text])
    mean_similarity = np.mean(similarity_by_caption)
    
    return mean_similarity, similarity_by_caption

if __name__ == '__main__':
    
    envfile = open(".env",'r')
    api_key = envfile.read()
    
    os.environ['OPENAI_API_KEY'] = api_key
    
    json_file = "/home/l1lyzhang/qwen-audio-generated-captions-audiocaps.json"
    f = open(json_file, 'r')
    caps_data = json.load(f)
    new_caps_data = json.loads(json.dumps(caps_data))
    f.close()
    
    client = OpenAI()
    
    for i in tqdm(range(len(caps_data))):
        ground_truth_captions = caps_data[i]["output"]
        generated_caption = caps_data[i]["generated"]
        similarity, similarity_by_caption = get_embedding_similarity_ada(client, ground_truth_captions, generated_caption)
        new_caps_data[i]['similarity'] = similarity
        new_caps_data[i]['similarity_by_caption'] = list(similarity_by_caption)
        if i % 500 == 0:
            generated_json = open("/home/l1lyzhang/qwen-audio-similarity-evaluation-audiocaps.json", 'w')
            json.dump(new_caps_data, generated_json, indent=4)
            generated_json.close()
    
    generated_json = open("/home/l1lyzhang/qwen-audio-similarity-evaluation-audiocaps.json", 'w')
    json.dump(new_caps_data, generated_json, indent=4)
    generated_json.close()