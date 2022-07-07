import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader 
from transformers import BertTokenizer 
import json 


class PDGMDataset(Dataset): 
    def __init__(self, data_path): 
        with open(data_path, 'r') as f: 
            self.all_data_pair = json.load(f)  
        self.tokenizer = BertTokenizer.from_pretrained('./ckpt/bert') 
        self.max_seq_len = 20 

    def __len__(self): 
        return len(self.all_data_pair) 

    def pad_tokens(self, index): 
        tokens = self.all_data_pair[index]['caption'] 
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
        tokens, mask = self.pad_tokens(index) 
        input_ids = torch.Tensor(self.all_data_pair[index]['image_token'][0]).long() 
        labels = torch.Tensor(self.all_data_pair[index]['image_token'][1]).long() 
        return tokens, mask, input_ids, labels 
        
