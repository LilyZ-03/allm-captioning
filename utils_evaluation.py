import json
import numpy as np
import matplotlib.pyplot as plt

def load_json(json_file):
    f = open(json_file, 'r')
    caps_data = json.load(f)
    f.close()
    return caps_data

def dump_json(json_data, new_file_name="new_json_file.json"):
    generated_json = open("/home/l1lyzhang/" + new_file_name, 'w')
    json.dump(json_data, generated_json, indent=4)
    generated_json.close()

def similarity_distribution(caps_data):

    cutoffs = [0.80, 0.85, 0.90, 0.95]
    counts = [0,0,0,0,0]
    
    for i in range(len(caps_data)):
        
        sim_val = caps_data[i]["similarity"]
        sim_val = np.max(caps_data[i]["similarity_by_caption"])
        
        if sim_val < 0.8:
            counts[0] += 1
        elif sim_val < 0.85:
            counts[1] += 1
        elif sim_val < 0.9:
            counts[2] += 1
        elif sim_val < 0.95:
            counts[3] += 1
        else:
            counts[4] += 1    
        
    return counts
    

    
if __name__ == '__main__':
    
    model_name = "salmonn"
    
    json_input = "/home/l1lyzhang/%s-similarity-evaluation-small.json" % model_name
    graph_dir = "/home/l1lyzhang/Evaluation/graphs/openai-ada/"
    graph_name = "%s-small-average-similarity-distribution.png" % model_name
    data = load_json(json_input)
    distribution = similarity_distribution(data)
    print(distribution)
    
    distribution = np.array(distribution)
    print(">=0.85: %f percent" % (np.sum(distribution[2:]) / np.sum(distribution) * 100))
    print(">=0.90: %f percent" % (np.sum(distribution[3:]) / np.sum(distribution) * 100))
    
    plt.bar(['<0.8','0.8-0.85','0.85-0.9','0.9-0.95','>0.95'],distribution)
    plt.title("%s-small, average-similarity-distribution" % model_name)
    # plt.savefig(graph_dir + graph_name)