import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import pickle
import os

import utils
from model import Condense
from transformers import BertTokenizer
from transformers import get_constant_schedule_with_warmup


def train(args):
  print(args)

  os.makedirs('model/%s/' % args.data_type, exist_ok=True)
  model_file = 'model/%s/condense.model' % args.data_type

  tokenizer = BertTokenizer.from_pretrained(args.bert_config)
  tokenizer.add_special_tokens({'additional_special_tokens': ['<movie>']})
  vocab_size = len(tokenizer)

  print('Loading datasets...')
  x_train, y_train = utils.condense_data(args.train_file, args.adjust_sentiment)
  if args.data_type == 'rotten':
    x_dev, y_dev = utils.condense_data(args.dev_file, args.adjust_sentiment)
  else:
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_dev = x_train[:2000]
    y_dev = y_train[:2000]
    x_train = x_train[2000:]
    y_train = y_train[2000:]

  print('Initializing models...')
  encoder = nn.Embedding(vocab_size, args.input_dim)
  encoder.cuda()

  model = Condense(args.aspect_dim, args.sentiment_dim, args.input_dim, args.hidden_dim, vocab_size)
  model.cuda()

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.998), eps=1e-9)
  scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup)

  best_loss = 10000
  if os.path.exists(model_file):
    print('Loading model checkpoint...')
    best_point = torch.load(model_file)
    encoder.load_state_dict(best_point['encoder'])
    model.load_state_dict(best_point['model'])
    optimizer.load_state_dict(best_point['optimizer'])
    best_loss = best_point['dev_loss']

  eval_at = args.evaluate_every
  stop_at = args.training_stopper

  step = 0
  print('Start training...')
  for epoch in range(args.num_epoch):
    if stop_at <= 0:
      break

    shuffle_indices = np.random.permutation(np.arange(len(x_train)))

    asp_losses = []
    asp_norm_losses = []
    sen_losses = []
    sen_norm_losses = []
    adv_losses = []

    train_iterator = tqdm(range(0, len(shuffle_indices), args.batch_size))
    for i in train_iterator:
      if stop_at <= 0:
        train_iterator.close()
        break
      if i+args.batch_size >= len(shuffle_indices):
        continue

      encoder.train()
      model.train()

      indices = shuffle_indices[i:i+args.batch_size]
      x_batch = [x_train[idx] for idx in indices]
      y_batch = [y_train[idx] for idx in indices]

      x_batch = [tokenizer.encode(x_inst) for x_inst in x_batch]
      x_batch, mask = utils.pad_text(x_batch)

      tokens = encoder(x_batch)
      before, after, sent_pred, adv_pred = model(tokens, mask, x_batch)

      sent_gold = torch.Tensor(y_batch).long().cuda()
      losses = model.calculate_loss(before, after, sent_pred, adv_pred, sent_gold)

      asp_losses.append(losses[0].item())
      asp_norm_losses.append(losses[1].item())
      sen_losses.append(losses[2].item())
      sen_norm_losses.append(losses[3].item())
      adv_losses.append(losses[4].item())

      batch_loss = torch.sum(torch.stack(losses))
      batch_loss.backward()
      nn.utils.clip_grad_norm_(encoder.parameters(), 2)
      nn.utils.clip_grad_norm_(model.parameters(), 2)
      nan_check = False
      for param in model.parameters():
        if param.grad is not None:
          if torch.isnan(param.grad.sum()):
            nan_check = True
            break
      if not nan_check:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

      eval_at -= len(x_batch)
      if eval_at <= 0:
        shuffle_indices = np.random.permutation(np.arange(len(x_dev)))
        x_dev = np.array(x_dev)[shuffle_indices]
        y_dev = np.array(y_dev)[shuffle_indices]

        train_asp_loss = np.mean(asp_losses)
        train_asp_norm_loss = np.mean(asp_norm_losses)
        train_sen_loss = np.mean(sen_losses)
        train_sen_norm_loss = np.mean(sen_norm_losses)
        train_adv_loss = np.mean(adv_losses)

        dev_asp_loss = []
        dev_asp_norm_loss = []
        dev_sen_loss = []
        dev_sen_norm_loss = []
        dev_adv_loss = []

        for j in tqdm(range(0, len(x_dev), args.batch_size)):
          encoder.eval()
          model.eval()

          x_batch = x_dev[j:j+args.batch_size]
          x_batch = [tokenizer.encode(x_inst) for x_inst in x_batch]
          x_batch, mask = utils.pad_text(x_batch)

          tokens = encoder(x_batch)
          before, after, sent_pred, adv_pred = model(tokens, mask, x_batch)

          sent_gold = torch.Tensor(y_dev[j:j+args.batch_size]).long().cuda()
          losses = model.calculate_loss(before, after, sent_pred, adv_pred, sent_gold)

          dev_asp_loss.append(losses[0].item())
          dev_asp_norm_loss.append(losses[1].item())
          dev_sen_loss.append(losses[2].item())
          dev_sen_norm_loss.append(losses[3].item())
          dev_adv_loss.append(losses[4].item())

        dev_asp_loss = np.mean(dev_asp_loss)
        dev_asp_norm_loss = np.mean(dev_asp_norm_loss)
        dev_sen_loss = np.mean(dev_sen_loss)
        dev_sen_norm_loss = np.mean(dev_sen_norm_loss)
        dev_adv_loss = np.mean(dev_adv_loss)
        dev_loss = dev_asp_loss + dev_asp_norm_loss + dev_sen_loss + dev_sen_norm_loss + dev_adv_loss

        tqdm.write("----------------------------------------------")
        tqdm.write("Epoch: %d, Batch: %d" % (epoch, i))
        tqdm.write("Train Losses: %.4f %.4f %.4f %.4f %.4f" % (train_asp_loss, train_asp_norm_loss, train_sen_loss, train_sen_norm_loss, train_adv_loss))
        tqdm.write("Dev Losses: %.4f %.4f %.4f %.4f %.4f" % (dev_asp_loss, dev_asp_norm_loss, dev_sen_loss, dev_sen_norm_loss, dev_adv_loss))

        if best_loss >= dev_loss:
          tqdm.write("UPDATING MODEL FILE...")
          best_loss = dev_loss
          stop_at = args.training_stopper
          torch.save({
            'encoder': encoder.state_dict(),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'dev_loss': dev_loss
          }, model_file)
        else:
          stop_at -= 1
          tqdm.write("STOPPING AT: %d" % stop_at)


        tqdm.write("----------------------------------------------")

        asp_losses = []
        asp_norm_losses = []
        sen_losses = []
        sen_norm_losses = []
        adv_losses = []
        eval_at = args.evaluate_every


def create_synthetic_data(args):
  print(args)

  file_name = 'data/%s/train.plan.json' % args.data_type

  alpha_a = args.alpha
  alpha_s = args.alpha

  condense_file = 'model/%s/condense.model' % args.data_type

  tokenizer = BertTokenizer.from_pretrained(args.bert_config)
  tokenizer.add_special_tokens({'additional_special_tokens': ['<movie>']})
  vocab_size = len(tokenizer)

  print('Loading corpus...')
  x_train, _ = utils.abstract_data(args.train_file, tokenizer)

  print('Loading models...')
  assert os.path.exists(condense_file)
  con_encoder = nn.Embedding(vocab_size, args.input_dim)
  con_encoder.requires_grad_(False)
  con_encoder.cuda()

  con_model = Condense(args.aspect_dim, args.sentiment_dim, args.input_dim, args.hidden_dim, vocab_size)
  con_model.requires_grad_(False)
  con_model.cuda()

  best_point = torch.load(condense_file)
  con_encoder.load_state_dict(best_point['encoder'])
  con_model.load_state_dict(best_point['model'])

  data = []
  vectors = []

  print('Creating synthetic dataset...')
  for i in tqdm(range(len(x_train))):
    x_batches = x_train[i]

    for x_idx in range(0, len(x_batches), 500):
      x_batch = x_batches[x_idx:x_idx+500]
      x_batch = [tokenizer.encode(x_inst) for x_inst in x_batch]
      if len(x_batch) < 100:
        continue

      token_ids, mask = utils.pad_text(x_batch)
      tokens = con_encoder(token_ids)
      _, doc, prob_a, prob_s = con_model.condense(tokens, mask)

      doc = doc.cpu().detach().numpy()
      prob_a = prob_a.cpu().detach().numpy() # b, a
      prob_s = prob_s.cpu().detach().numpy() # b, s

      for idx, (d, a, s) in enumerate(zip(doc, prob_a, prob_s)):
        if not utils.check_summary_worthy(x_batch[idx], tokenizer, 
                                          args.min_length, args.max_length, args.max_symbols, args.max_tridots):
          continue

        N = -1
        while N < args.min_reviews or N > min(len(x_batch), args.max_reviews):
          N = np.random.normal(args.mean_reviews, args.std_reviews)
        N = int(N)

        a_ = np.random.dirichlet(alpha_a*a+1e-9, N)[:,np.newaxis] # N, a
        s_ = np.random.dirichlet(alpha_s*s+1e-9, N)[:,np.newaxis] # N, s

        dist_a = np.sqrt(((np.sqrt(prob_a[np.newaxis]) - np.sqrt(a_))**2).sum(-1))
        dist_s = np.sqrt(((np.sqrt(prob_s[np.newaxis]) - np.sqrt(s_))**2).sum(-1))

        dist = dist_a + dist_s
        dist[:,idx] = 1e9

        idx_set = []
        for d in dist:
          d = np.argsort(d)
          for d_ in d:
            if d_ not in idx_set:
              idx_set.append(d_)
              break

        inst = {}
        inst['summary'] = ' '.join(tokenizer.decode(x_batch[idx]).split()[1:-1])
        inst['reviews'] = [' '.join(tokenizer.decode(x_batch[i]).split()[1:-1]) for i in idx_set if idx != i]
        data.append(inst)

  f = open(file_name, 'w')
  json.dump(data, f, indent=2)
  f.close()
  print('Dataset saved.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('-mode', default='train', type=str) # train or create

  parser.add_argument('-data_type', default='yelp', type=str) # rotten, yelp, or amazon

  parser.add_argument('-aspect_dim', default=100, type=int) # 50 (rotten), 100 (yelp or amazon)
  parser.add_argument('-sentiment_dim', default=5, type=int) # 2 (rotten), 5 (yelp or amazon)
  parser.add_argument('-adjust_sentiment', default=1, type=int) # 0 (rotten), 1 (yelp or amazon)

  parser.add_argument('-input_dim', default=256, type=int)
  parser.add_argument('-hidden_dim', default=256, type=int)

  parser.add_argument('-num_epoch', default=30, type=int)
  parser.add_argument('-batch_size', default=16, type=int)
  parser.add_argument('-learning_rate', default=3e-5, type=float)

  parser.add_argument('-warmup', default=8000, type=int)
  parser.add_argument('-evaluate_every', default=20000, type=int)
  parser.add_argument('-training_stopper', default=50, type=int)

  parser.add_argument('-train_file', default='data/yelp/train.json', type=str)
  parser.add_argument('-dev_file', default='data/yelp/dev.json', type=str)
  parser.add_argument('-bert_config', default='bert-base-uncased', type=str)

  # used when creating synthetic dataset
  parser.add_argument('-alpha', default=10.0, type=float) # 1.0 (rotten), 10.0 (yelp or amazon)

  parser.add_argument('-min_length', default=50, type=int) # 10 (rotten), 50 (yelp or amazon)
  parser.add_argument('-max_length', default=90, type=int) # 10 (rotten), 90 (yelp), 1000 (amazon)
  parser.add_argument('-max_symbols', default=0, type=int) # 0 (rotten or yelp), 1000 (amazon)
  parser.add_argument('-max_tridots', default=0, type=int) # 0 (rotten or yelp), 1000 (amazon)

  parser.add_argument('-min_reviews', default=8, type=int) # 10 (rotten), 8 (yelp or amazon)
  parser.add_argument('-max_reviews', default=8, type=int) # 160 (rotten), 8 (yelp or amazon)
  parser.add_argument('-mean_reviews', default=8, type=int) # 100 (rotten), 8 (yelp or amazon)
  parser.add_argument('-std_reviews', default=0, type=int) # 60 (rotten), 8 (yelp or amazon)

  args = parser.parse_args()
  if args.mode == 'train':
    train(args)
  elif args.mode == 'create':
    create_synthetic_data(args)