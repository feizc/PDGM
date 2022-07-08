# Incorporating deepspeed 
# Referring to: https://www.deepspeed.ai/tutorials/cifar-10/ 
import torch 
import argparse 
import deepspeed 
from tqdm import tqdm 
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import functional as nnf 
from dataset import PDGMDataset
from PDGM import TextEncoderConfig, ProgressiveDecoderConfig, PDGModel 
from torch.utils.data import DataLoader



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


def train(train_dataloader, model_engine, optimizer, args): 
    
    for epoch in range(args.epochs):
        loss_sum = 0 
        acc_sum = 0 
        progress = tqdm(total=len(train_dataloader), desc='PDGM')
        for idx, (tokens, mask, image_ids, labels) in enumerate(train_dataloader): 

            tokens, mask, image_ids, labels = tokens.to(model_engine.device), mask.to(model_engine.device), image_ids.to(model_engine.device), labels.to(model_engine.device)
            logits = model_engine(text_ids=tokens, text_attention_mask=mask, image_ids=image_ids) 
            loss = reweight_cross_entropy(logits.reshape(-1, logits.shape[-1]), image_ids.flatten(), labels.flatten()) 

            model_engine.backward(loss) 
            model_engine.step()
            loss_sum += loss.item()
            predicts = torch.argmax(logits, dim=-1).flatten() 
            acc = torch.eq(predicts.cpu(), labels.flatten().cpu()).float().sum() / predicts.size(0) 
            acc_sum += acc.item()
            progress.set_postfix({"loss": loss_sum / (idx + 1), "acc": acc_sum / (idx + 1)})
            progress.update()
            break 
        progress.close() 
        break


def add_argument(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--with_cuda', default=True, action='store_true',
                         help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                         help='whether use exponential moving average')

    # train
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                         help='mini-batch size (default: 2)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                         help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

     # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    return args


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
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    text_config = TextEncoderConfig()
    image_config = ProgressiveDecoderConfig() 
    model = PDGModel(text_config=text_config, image_config=image_config) 

    ds_args = add_argument() 
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(args=ds_args, model=model, model_parameters=parameters, training_data=train_dataset)

    train(train_dataloader, model_engine, optimizer, args) 



if __name__ == '__main__':
    main() 