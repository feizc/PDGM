# Construct training pair data for PDGM 
import torch 
import os 
import PIL 
import json 
import pickle 

import torch 
import torchvision.transforms as T 
import torchvision.transforms.functional as TF 
import torch.nn.functional as F 
from tqdm import tqdm 

from case import preprocess 
from dall_e import map_pixels, unmap_pixels, load_model 


image_token_len = 256


def data_pair_generation(probability, latents, step_num=16): 
    k_num = int(image_token_len / step_num) 
    probability = probability.view(-1) 
    latents = latents.view(-1) 
    data_pair = [] 
    data_pair.append(latents.tolist())
    for i in range(1, step_num+2): 
        threshold, _ = torch.kthvalue(probability, k_num*i) 
        latents[probability <= threshold] = 8192 # pad idx 
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

    data = data['annotations'] 
    f = open(out_path, 'wb')
    for i in tqdm(range(len(data))): 
        d = data[i] 
        caption = d['caption']
        img_id = d['image_id'] 
        file_name = os.path.join(data_path, f'train2014/COCO_train2014_{int(img_id):012d}.jpg') 
        img = PIL.Image.open(file_name) 
        img = img.convert('RGB') 
        x = preprocess(img).to(device)
        z_logits = enc(x) 
        p, z = torch.max(z_logits, dim=1)  
        data_pair = data_pair_generation(p, z)
        for j in range(len(data_pair)-2): 
            item = {
                'caption': caption, 
                'image_token': [data_pair[j+1], data_pair[j]], 
            }
            pickle.dump(item, f)
        break
    f.close()


def pkl_check(pickle_path):
    f = open(pickle_path, 'rb') 
    num = 0
    while True: 
        try: 
            item = pickle.load(f) 
            """
            t = 0
            for i in item['image_token'][0]:
                if i == 8192:
                    t += 1
            print(t)
            t = 0
            for i in item['image_token'][1]:
                if i == 8192:
                    t += 1
            print(t)
            print('one step')
            """
            num += 1
        except EOFError:
            break
    print(num)

 

if __name__ == "__main__":
    data_path =  '/Users/feizhengcong/Desktop/COCO' 
    out_path = './data/train.pkl' 
    data_preprocess(data_path, out_path) 
    pkl_check(out_path)
