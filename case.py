# Plot the figure according to the confidence of VQ-VAE 
import io 
import os, sys 
import PIL
from matplotlib.pyplot import axis 

import torch 
import torchvision.transforms as T 
import torchvision.transforms.functional as TF 
import torch.nn.functional as F 
from matplotlib import pyplot as plt

from dall_e import map_pixels, unmap_pixels, load_model 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
target_image_size = 256


def preprocess(img):
    s = min(img.size)
    
    if s < target_image_size: 
        r = target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        # img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.resize(img, s, interpolation=T.InterpolationMode.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [target_image_size]) 
    
    else: 
        img = TF.resize(img, (target_image_size, target_image_size), interpolation=T.InterpolationMode.LANCZOS)
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    
    return map_pixels(img) 


def main(): 
    # load vq-vae model 
    enc = load_model('./ckpt/vqvae/encoder.pkl', device) 
    dec = load_model('./ckpt/vqvae/decoder.pkl', device) 
    
    img_path = './data/case.jpg' 
    img = PIL.Image.open(img_path)
    img.show()

    x = preprocess(img) 
    z_logits = enc(x) # (bsz, vocab=8192, 32, 32)
    # z = torch.argmax(z_logits, axis=1) 
    p, z = torch.max(z_logits, dim=1)
    for i in range(1, 17): 
        p = p.view(-1) 
        threshold, _ = torch.kthvalue(p, 64*i) # top-k small 
        p = p.view(1, 32, 32)
        z[p < threshold] = 0
        
        z_one = F.one_hot(z, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float() 

        x_stats = dec(z_one).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        x_rec = T.ToPILImage(mode='RGB')(x_rec[0]) 
        plt.subplot(2, 8, i)
        plt.xticks([]) 
        plt.yticks([])
        plt.imshow(x_rec)
        #x_rec.show() 
        print(i) 
    #plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main() 
