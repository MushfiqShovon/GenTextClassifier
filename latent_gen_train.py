import sys
import os
import time
import random
import shutil
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import ModelBase
import utils_data
from utils_train import resume_checkpoint, save_checkpoint, parse_args

def train_epoch(traindata, trainlabel, validdata, validlabel, model, criterion, optimizer, args, epoch):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch_ind, sents_label in enumerate(utils_data.batches(traindata, trainlabel, args.batch_size, args.cuda)):

        sents, labels = sents_label

        batch_ind += 1
        optimizer.zero_grad()
        hidden = model.init_hidden(len(sents))

        losses = model(sents, labels, hidden, criterion, False, False)

        losses = torch.sum(losses, dim=0)

        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += losses.item()

        if batch_ind % args.log_interval == 0:
            elapsed = time.time() - start_time
            _, train_acc = evaluate(sents, labels, model, criterion, args)
            val_loss, val_acc = evaluate(validdata, validlabel, model, criterion, args)

            logging.info('| epoch %d | %5d/%5d batches | lr %5.5f | ms/batch %5.2f | '
                    'train loss %5.2f | train acc %5.2f | valid loss %.2f | valid acc %.2f',
                epoch, batch_ind, len(trainlabel) // args.batch_size, args.lr,
                elapsed * 1000 / args.log_interval, losses.item(), train_acc,
                val_loss, val_acc)
            model.train()

    return total_loss / batch_ind

def evaluate(validdata, validlabel, model, criterion, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_correct = 0.
    with torch.no_grad():
        for sents, labels in utils_data.batches(validdata, validlabel, args.batch_size, args.cuda):
            losses = []
            for y in range(args.nclass):
                hidden = model.init_hidden(len(sents))
                y_s = torch.LongTensor([y] *  len(sents))
                loss = model(sents, y_s, hidden, criterion, True, False)
                losses.append(loss)
            losses = torch.cat(losses, dim=0).view(-1, len(sents))
            prediction = torch.argmin(losses, dim=0)

            num_correct = (prediction == labels).float().sum()
            total_loss += torch.sum(torch.min(losses, dim=0)[0]).item()
            total_correct += num_correct.item()

    return total_loss / len(validlabel), \
        total_correct / len(validlabel) * 100.0

def main(args=sys.argv[1:]):
    args = parse_args(args)
    args.nclass = len(open(os.path.join('data', args.dataset, 'classes.txt'), 'r').readlines())

    logging.basicConfig(level=args.logging)
    logging.info('Args: %s', args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # load data
    data_dict = torch.load(os.path.join('data', args.dataset, 'data', args.datafile))
    traindata = data_dict['traindata']
    trainlabel = data_dict['trainlabel']
    validdata = data_dict['validdata']
    validlabel = data_dict['validlabel']
    testdata = data_dict['testdata']
    testlabel = data_dict['testlabel']
    args.vocab_size = data_dict['vocabsize']

    model = ModelBase.get_model(args.mode, args.vocab_size, args.word_emb_dim, args.label_emb_dim, 
        args.cond_emb_dim, args.hid_dim, args.nlayers, args.nclass, args.ncondition, args.dropout, 
        args.use_EM, args.infer_method, args.cuda, args.one_hot, args.bilinear)
    criterion = nn.CrossEntropyLoss(reduce=False)

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else 
        x.size()[0] for x in params if x.size())
    logging.info('Model total parameters: %d', total_params)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, 
                                    weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, 
                                    weight_decay=args.wdecay)

    args.start_epoch = 0
    args.best_val_acc = None

    if args.resume:
        args, model, optimizer = resume_checkpoint(args, model, optimizer, logging)
    
    try:
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            logging.info("Training epoch %d", epoch)

            train_loss = train_epoch(traindata, trainlabel, validdata, validlabel, model, 
                criterion, optimizer, args, epoch)

            logging.info('-' * 89)
            logging.info('| end of epoch %3d | time: %.2f s | train loss %5.2f ', 
                epoch, (time.time() - epoch_start_time), train_loss)
            logging.info('-' * 89)

            if (epoch+1) % args.save_interval == 0:
                eval_start_time = time.time()
                val_loss, val_acc = evaluate(validdata, validlabel, model, criterion, args)

                logging.info('-' * 89)
                logging.info('| time: %.2f s | valid loss %.2f | valid acc %.2f', 
                    (time.time() - eval_start_time), val_loss, val_acc)
                logging.info('-' * 89)

                is_best = not args.best_val_acc or val_acc > args.best_val_acc
                if is_best:
                    args.best_val_acc = val_acc
                    args.best_epoch = epoch

                filename = args.save_prefix + '_' + str(args.save_interval) + '.chkp'

                save_checkpoint({
                    'args': args,
                    'state_dict': model.state_dict(),
                    'best_val_acc': args.best_val_acc,
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                }, args.dataset, filename)
                
                if is_best:
                    shutil.copyfile(os.path.join('experiment', args.dataset, filename), 
                        os.path.join('experiment', args.dataset, args.save_prefix + '_best.chkp'))
    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early')

    # test
    checkpoint = torch.load(os.path.join('experiment', args.dataset, args.save_prefix + '_best.chkp'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()
    test_loss, test_acc = evaluate(testdata, testlabel, model, criterion, args)
    logging.info('=' * 89)
    logging.info('| Test | test loss %.2f | test acc %.2f', test_loss, test_acc)
    logging.info('=' * 89)

if __name__ == '__main__':
    main()