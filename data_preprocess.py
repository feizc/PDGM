# Construct training pair data for PDGM 
import torch 
import os 
import PIL 
import json 

import torch 
import torchvision.transforms as T 
import torchvision.transforms.functional as TF 
import torch.nn.functional as F 
from tqdm import tqdm 

from case import preprocess 
from dall_e import map_pixels, unmap_pixels, load_model 



def data_pair_generation(probability, latents, step_num=16): 
    k_num = int(1024 / step_num) 
    probability = probability.view(-1) 
    latents = latents.view(-1) 
    data_pair = [] 
    data_pair.append(latents.tolist())
    for i in range(1, step_num+1): 
        threshold, _ = torch.kthvalue(probability, 64*i) 
        latents[probability < threshold] = 8192 # pad idx 
        data_pair.append(latents.tolist()) 
    return data_pair 



def data_preprocess(data_path, out_path, step_num=16):
    """
    out_path: the output path for preprocessed training data 
    step_num: cut the image token sequence into step_num stages 
    """ 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    # load vq-vae model 
    enc = load_model('./ckpt/vqvae/encoder.pkl', device) 
    dec = load_model('./ckpt/vqvae/decoder.pkl', device)  

    with open(os.path.join(data_path, 'annotations/captions_train2014.json'), 'r') as f: 
        data = json.load(f) 

    all_data_pair = [] 
    data = data['annotations'] 
    for i in tqdm(range(len(data))): 
        d = data[i] 
        caption = d['caption']
        img_id = d['image_id'] 
        file_name = os.path.join(data_path, f'train2014/COCO_train2014_{int(img_id):012d}.jpg') 
        img = PIL.Image.open(file_name) 
        x = preprocess(img).to(device)
        z_logits = enc(x) 
        p, z = torch.max(z_logits, dim=1)  
        data_pair = data_pair_generation(p, z)
        for j in range(len(data_pair)-2): 
            item = {
                'caption': caption, 
                'image_token': [data_pair[j+1], data_pair[j]], 
            }
            all_data_pair.append(item) 
        if (i + 1) % 10000 == 0: 
            with open(out_path, 'w') as f: 
                json.dump(all_data_pair, f, indent=4)
    
    with open(out_path, 'w') as f: 
        json.dump(all_data_pair, f, indent=4)



if __name__ == "__main__":
    data_path =  '/Users/feizhengcong/Desktop/COCO' 
    out_path = './data/train.json' 
    data_preprocess(data_path, out_path) 