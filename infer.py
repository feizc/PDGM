import torch 
import torch.nn.functional as F 
import torchvision.transforms as T 
import json 

from PDGM import TextEncoderConfig, ProgressiveDecoderConfig, PDGModel  
from transformers import BertTokenizer, data 
from dall_e import load_model, unmap_pixels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
step_num = 16 


def text_preprocess(text, tokenizer, max_seq_len=30): 
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.int64) 
    padding = max_seq_len - tokens.shape[0]
    if padding > 0:
        tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        tokens = tokens[:max_seq_len]
    mask = tokens.ge(0)  # mask is zero where we out of sequence
    tokens[~mask] = 0
    mask = mask.float() 
    return tokens, mask 


def image_save(latents, image_dec, i): 
    z = torch.where(latents < 8192, latents, 0)  
    z = z.view(1, 32, 32) 
    z_one = F.one_hot(z, num_classes=8192).permute(0, 3, 1, 2).float() 

    x_stats = image_dec(z_one).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    x_rec = T.ToPILImage(mode='RGB')(x_rec[0]) 
    x_rec.save(str(i)+'.jpg')


def condition_image_generation(text, tokenizer, model, image_dec): 
    tokens, mask = text_preprocess(text, tokenizer) 
    image_ids = torch.Tensor([8192] * 1024).long() 
    for i in range(step_num): 
        t_tokens, t_mask, image_ids = tokens.to(device).unsqueeze(0), mask.to(device).unsqueeze(0), image_ids.to(device).unsqueeze(0) 
        logits = model(text_ids=t_tokens, text_attention_mask=t_mask, image_ids=image_ids)[0] 
        z = torch.argmax(logits, axis=1) 
        image_save(z, image_dec, i)
        image_ids = z 
        break 
    


def create_from_json(path): 
    image_dec = load_model('./ckpt/vqvae/decoder.pkl', device) 
    with open(path, 'r') as f: 
        data_list = json.load(f)
    for i in range(len(data_list)): 
        z = torch.Tensor(data_list[i]).long() 
        print(z) 
        image_save(z, image_dec, i)



def main(): 
    text_config = TextEncoderConfig()
    image_config = ProgressiveDecoderConfig() 
    model = PDGModel(text_config=text_config, image_config=image_config) 
    
    model_path = './ckpt/latest.pth' 
    model.load_state_dict(torch.load(model_path)) 

    model = model.to(device) 
    tokenizer = BertTokenizer.from_pretrained('./ckpt/bert') 
    image_dec = load_model('./ckpt/vqvae/decoder.pkl', device) 
    
    text = 'A very clean and well decorated empty bathroom'
    condition_image_generation(text, tokenizer, model, image_dec) 




if __name__ == '__main__':
    # main() 
    path = './result/result.json' 
    create_from_json(path) 