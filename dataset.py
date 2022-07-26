import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader 
from transformers import BertTokenizer 
import json 
import os 
import PIL 
import pickle

from case import preprocess 


class FastPDGMDataset(Dataset): 
    "Load data from pre-extracted features with pickle"
    def __init__(self, data_path): 
        self.data_path = data_path 
        self.f = open(data_path, 'rb')
        self.tokenizer = BertTokenizer.from_pretrained('./ckpt/bert') 
        self.max_seq_len = 20 

    def __len__(self): 
        return 16

    def pad_tokens(self, tokens): 
        tokens = torch.tensor(self.tokenizer.encode(tokens), dtype=torch.int64) 
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float() 
        return tokens, mask 
    
    def __getitem__(self, index): 
        try: 
            item = pickle.load(self.f) 
        except EOFError:
            self.f = open(self.data_path, 'rb') 
            item = pickle.load(self.f) 
        tokens, mask = self.pad_tokens(item['caption']) 
        input_ids = torch.Tensor(item['image_token'][0]).long() 
        labels = torch.Tensor(item['image_token'][1]).long() 
        return tokens, mask, input_ids, labels 
        



class PDGMDataset(Dataset): 
    def __init__(self, data_path, vqvae_encoder, device): 
        self.data_path = data_path 
        with open(os.path.join(data_path, 'annotations/captions_train2014.json'), 'r') as f: 
            self.data = json.load(f)['annotations'] 
        self.tokenizer = BertTokenizer.from_pretrained('./ckpt/bert') 
        self.max_seq_len = 20 
        self.vqvae_encoder = vqvae_encoder 
        self.device = device 
        self.step_num = 16 
    
    def __len__(self): 
        return len(self.data) * self.step_num  

    def pad_tokens(self, index): 
        tokens = self.data[index]['caption'] 
        tokens = torch.tensor(self.tokenizer.encode(tokens), dtype=torch.int64) 
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float() 
        return tokens, mask 
         
    
    def __getitem__(self, i): 
        index = int(i / self.step_num) 
        step = int(i % self.step_num)

        tokens, mask = self.pad_tokens(index) 

        img_id = self.data[index]['image_id'] 
        file_name = os.path.join(self.data_path, f'train2014/COCO_train2014_{int(img_id):012d}.jpg') 
        img = PIL.Image.open(file_name) 
        img = img.convert('RGB') 
        x = preprocess(img).to(self.device) # (1, 3, 256, 256) 
        z_logits = self.vqvae_encoder(x) 
        p, z = torch.max(z_logits, dim=1)  
        p = p.view(-1) 
        z = z.view(-1) 

        if step == 0: 
            threshold = torch.min(p) - 1 
        else:
            threshold, _ = torch.kthvalue(p, 64*step) 
        z[p <= threshold] = 8192 
        labels = z.clone()
        
        threshold, _ = torch.kthvalue(p, 64*(step+1)) 
        z[p <= threshold] = 8192 
        input_ids = z.clone()

        return tokens, mask, input_ids, labels 

