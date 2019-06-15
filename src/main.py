# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn

import data
import model
import os
import os.path as osp
from loss import Perplexity

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used')
parser.add_argument('--save', type=str, default='model_{}.pt'.format(time.strftime('%Y%m%d-%H-%M-%S')),
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = args.cuda

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

    
tie_weights = args.tied
lr=20
# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)

# WRITE CODE HERE within two '#' bar
########################################
# Build LMModel model (bulid your language model here)\
ntokens = len(data_loader.word_id)
model=model.LMModel(nvoc=len(data_loader.word_id),ninput=300, nhid=32, nlayers=1, tie_weights).to(device)

########################################

criterion = nn.CrossEntropyLoss()


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function

# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.
def evaluate():
    # Turn on evaluation mode which disables dropout.
    data_loader.set_valid()
    model.eval()
    total_loss = 0.
    ntokens = len(data_loader.word_id)
    hidden = model.init_hidden(eval_batch_size)
    end_flag = False
    with torch.no_grad():
        while not end_flag:
            #data, targets = get_batch(data_source, i)
            data, targets, end_flag= data_loader.get_batch()
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_loader.valid) - 1)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train():
    # Turn on training mode which enables dropout.
    data_loader.set_train()
    model.train()
    total_loss = 0.
    batch = 0
    start_time = time.time()
    ntokens = len(data_loader.word_id)
    hidden = model.init_hidden(train_batch_size)
    end_flag = False
    while not end_flag:
        data, targets, end_flag= data_loader.get_batch()
        #data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()
        log_interval=200
        batch+=1
        if batch % log_interval == 0 and batch > 0:
            
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(data_loader.train) // args.max_sql, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()




# Loop over epochs.
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate()
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f} |'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    print('-' * 89)

    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
        lr /= 1.15
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 3

