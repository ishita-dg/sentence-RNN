import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

import numpy as np
import itertools

"""
RNN model trained on sentences. Base code from pytorch examples word_language_model.
"""

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/Wiki RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=20,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=20,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=15,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=3, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=300,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=1.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
print("\nLoading...\n")
#corpus_words = data.Corpus(args.data, to_sentence = False)
corpus = data.Corpus(args.data, to_sentence = True)

def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), 
                                                 *tensor.size()[1:]).zero_()])

def batch_sents(data, bsz, nsen = 1):
    """
    Returns a dictionary of batched inputs, targets and 
    sequence lengths of sentences in each batch
    inp: [max(seq_lens), nbatch]
    tar: [sum(seq_lens)]
    """
    # number of sentences, and number of chunks to be batched
    N = len(data)
    nch = N + 1 - nsen
    nbatch = nch // bsz
    nch_batch = nbatch*bsz
    
    data_c = []
    for i in np.arange(0, nch_batch, 1):
        data_c.append(torch.cat((data[i:i+nsen]), 0))
    
    batches = []
    for i in np.arange(0, nch_batch, bsz):
        batch = {}
        sens = data_c[i:i+bsz]
        lsens = [len(ch) for ch in sens]
        
        l2s = np.argsort(-np.array(lsens))
        
        o_lsens = [lsens[i] for i in l2s]
        o_sens = [sens[i] for i in l2s]
        
        max_length = o_lsens[0]
        #batch_sizes = [sum(map(bool, filter(lambda x: x >= i, o_lsens))) for i in range(1, max_length + 1)]
        padded_inp = torch.cat([pad(o_sens[i][:-1].view(o_lsens[i]-1, 1), 
                                    max_length-1).view(-1,1) 
                                for i in range(0, bsz, 1)], 1)
        
        tar = torch.Tensor([inner for outer in o_sens for inner in outer[1:]]).long()
        
        padded_inp = Variable(torch.squeeze(padded_inp.long()))
        tar = Variable(torch.squeeze(tar))
    
        batch["lens"] = [x -1 for x in o_lsens]
        batch["inp"] = padded_inp
        batch["tar"] = tar        
        
        batches.append(batch)
        
    return batches

eval_batch_size = 10

train_data = batch_sents(corpus.train, args.batch_size)
val_data = batch_sents(corpus.valid, eval_batch_size)
test_data = batch_sents(corpus.test, eval_batch_size)

##############################################################################
# Build the model
###############################################################################

print("\nBuilding...\n")

ntokens = len(corpus.dictionary)
#model_words = model.RNNModel_words(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

model = model.RNNModel_sents(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)

if args.cuda:
    model.cuda()

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################
print("\nTraining...\n")

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    for i, batch in enumerate(data_source):
        hidden = model.init_hidden(eval_batch_size)
        data, targets, seq = batch['inp'], batch['tar'], batch['lens']
        output, hidden = model(data, seq, hidden)
        total_loss += criterion(output.view(-1, ntokens), 
                                targets.view(-1)).data
        #hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    #hidden = model.init_hidden(args.batch_size)
    count = 0
    for i, batch in enumerate(train_data):
        data, targets, seq = batch['inp'], batch['tar'], batch['lens']
        hidden = model.init_hidden(args.batch_size)
        model.zero_grad()
        output, hidden = model(data, seq, hidden)
        loss = criterion(output.view(-1, ntokens), targets.view(-1))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            print("\nSaved with validation loss = {:0.2f}".format(best_val_loss))
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 2.0
            print("\nAnnealed LR to {:0.4f}".format(lr))
            
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
print("\nTesting...\n")
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f}'.format(test_loss))
print('=' * 89)
