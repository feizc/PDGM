import torch 
import argparse 
from tqdm import tqdm 
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import functional as nnf 
from dataset import PDGMDataset
from PDGM import TextEncoderConfig, ProgressiveDecoderConfig, PDGModel 
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


def reweight_cross_entropy(predict, input, target): 
    index = torch.eq(input, target)
    ones = torch.ones_like(target)
    weight = torch.where(index, ones, 10) 
    
    exp = torch.exp(predict) 
    tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze() 
    tmp2 = exp.sum() 
    softmax = tmp1 / tmp2 
    log = -torch.log(softmax)  
    loss = log * weight 
    return loss.mean()


def train(train_dataloader, model, args): 
    model.train() 
    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warm_up, num_training_steps=args.epochs * len(train_dataloader)
    ) 
    for epoch in range(args.epochs):
        loss_sum = 0 
        acc_sum = 0 
        progress = tqdm(total=len(train_dataloader), desc='PDGM')
        for idx, (tokens, mask, image_ids, labels) in enumerate(train_dataloader): 
            model.zero_grad() 
            tokens, mask, image_ids, labels = tokens.to(device), mask.to(device), image_ids.to(device), labels.to(device)
            logits = model(text_ids=tokens, text_attention_mask=mask, image_ids=image_ids) 
            #loss1 = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.flatten(), ignore_index=8192) 
            #loss2 = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.flatten())
            # loss = args.alpha * loss1 + loss2 
            loss = reweight_cross_entropy(logits.reshape(-1, logits.shape[-1]), image_ids.flatten(), labels.flatten()) 

            loss.backward() 
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad() 
            loss_sum += loss.item()
            predicts = torch.argmax(logits, dim=-1).flatten() 
            acc = torch.eq(predicts.cpu(), labels.flatten().cpu()).float().sum() / predicts.size(0) 
            acc_sum += acc.item()
            progress.set_postfix({"loss": loss_sum / (idx + 1), "acc": acc_sum / (idx + 1)})
            progress.update()
            break 
        progress.close()
        torch.save(
            model.state_dict(), 
            './ckpt/latest.pth', 
        )
        break



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', default='./data/train.json') 
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warm_up', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--alpha', type=int, default=10)
    args = parser.parse_args() 

    train_dataset = PDGMDataset(args.data_path) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    text_config = TextEncoderConfig()
    image_config = ProgressiveDecoderConfig() 
    model = PDGModel(text_config=text_config, image_config=image_config) 
    model = model.to(device) 

    train(train_dataloader, model, args) 



if __name__ == '__main__':
    main() 