import torch 
import argparse 
from tqdm import tqdm 
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import functional as nnf 
from dataset import PDGMDataset
from PDGM import TextEncoderConfig, ProgressiveDecoderConfig, PDGModel 
from torch.utils.data import DataLoader
from dall_e import load_model 


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


def pre_train(train_dataloader, model, args): 
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



# Optimize with mutual information 
def fine_tune(train_dataloader, model, args, alpha=0.1): 
    model.train() 
    optimizer = AdamW(model.parameters(), lr=args.ft_lr)  
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warm_up, num_training_steps=args.ft_epochs * len(train_dataloader)
    ) 
    for epoch in range(args.ft_epochs): 
        loss_sum = 0 
        acc_sum = 0 
        progress = tqdm(total=len(train_dataloader), desc='PDGM ft') 
        for idx, (tokens, mask, image_ids, labels) in enumerate(train_dataloader): 
            model.zero_grad() 
            tokens, mask, image_ids, labels = tokens.to(device), mask.to(device), image_ids.to(device), labels.to(device) 
            uncondi_tokens = torch.zeros_like(tokens).to(device) 

            logits = model(text_ids=tokens, text_attention_mask=mask, image_ids=image_ids) 
            uncondi_logits = model(text_ids=uncondi_tokens, text_attention_mask=mask, image_ids=image_ids) 
            logits_cb = ((1 - alpha) * logits + alpha * uncondi_logits).log_softmax(2).detach() 

            kl = -((logits - logits_cb) * logits_cb.exp()).sum(2).mean() 
            kl.backward() 
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad() 
            loss_sum += kl.item()
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
    parser.add_argument('--data_path', default='/Users/feizhengcong/Desktop/COCO' ) 
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ft_lr', type=float, default=1e-6)
    parser.add_argument('--warm_up', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=20) 
    parser.add_argument('--ft_epochs', type=int, default=5)
    parser.add_argument('--alpha', type=int, default=10)
    args = parser.parse_args() 

    vqvae_decoder = load_model('./ckpt/vqvae/encoder.pkl', device)  
    train_dataset = PDGMDataset(args.data_path, vqvae_decoder, device) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    text_config = TextEncoderConfig()
    image_config = ProgressiveDecoderConfig() 
    model = PDGModel(text_config=text_config, image_config=image_config) 
    model = model.to(device) 

    pre_train(train_dataloader, model, args) 
    fine_tune(train_dataloader, model, args)




if __name__ == '__main__':
    main() 