import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import pickle
import os

import utils
from model import Condense, Abstract
from transformers import BertTokenizer, BertForMaskedLM
from transformers import get_constant_schedule_with_warmup


def train(args):
  print(args)

  condense_file = 'model/%s/condense.model' % args.data_type
  abstract_file = 'model/%s/abstract.model' % args.data_type

  tokenizer = BertTokenizer.from_pretrained(args.bert_config)
  tokenizer.add_special_tokens({'additional_special_tokens': ['<movie>']})
  vocab_size = len(tokenizer)

  print('Loading datasets...')
  x_train, y_train = utils.abstract_data(args.train_file)
  x_dev, y_dev = utils.abstract_data(args.test_file, multi_ref=args.multi_ref)

  print('Initializing models...')
  language_model = BertForMaskedLM.from_pretrained(args.bert_config)
  language_model.requires_grad_(False)
  language_model.cuda()

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

  model = Abstract(vocab_size, args.hidden_dim, args.hidden_dim)
  model.cuda()

  optimizer = torch.optim.Adam(model.parameters())

  best_acc = 0
  saved_models = []
  if os.path.exists(abstract_file):
    print('Loading model checkpoint...')
    best_point = torch.load(abstract_file)
    model.load_state_dict(best_point['model'])
    optimizer.load_state_dict(best_point['optimizer'])
    best_acc = best_point['dev_acc']

  eval_at = args.evaluate_every
  stop_at = args.training_stopper

  losses = []
  gate = []

  print('Start training...')
  for epoch in range(args.num_epoch):
    if stop_at <= 0:
      break

    shuffle_indices = np.random.permutation(len(x_train))

    for step in tqdm(range(0, len(x_train), args.batch_size)):
      if stop_at <= 0:
        break

      indices = shuffle_indices[step:step+args.batch_size]
      x_batch = [x_train[idx] for idx in indices]
      y_batch = [y_train[idx] for idx in indices]

      x_batch = [[tokenizer.encode(x_rev) for x_rev in x_inst] for x_inst in x_batch]
      y_batch = [tokenizer.encode(y_inst) for y_inst in y_batch]

      model.train()

      tokens_batch, token_ids_batch, aspect_batch, sentiment_batch = utils.run_condense(x_batch, tokenizer, con_encoder, con_model)
      output_smooth_batch, output_mask_batch = utils.bert_label_smoothing(y_batch, tokenizer, language_model)

      tokens_batch = utils.pad_vector(tokens_batch, args.hidden_dim)[0]
      token_ids_batch, token_mask_batch = utils.pad_text(token_ids_batch)
      aspect_batch = torch.Tensor(aspect_batch).float().cuda() # batch size, hidden dim
      sentiment_batch = torch.Tensor(sentiment_batch).float().cuda() # batch size, hidden dim

      output_batch, _ = utils.pad_text(y_batch)

      _, gt, loss = model(tokens_batch, token_ids_batch, token_mask_batch,
                          aspect_batch, sentiment_batch, 
                          output_batch, output_smooth_batch, output_mask_batch)
      losses.append(loss.item())
      gate.append(gt.mean().item())

      try:
        loss.backward()
      except:
        continue
      nn.utils.clip_grad_norm_(model.parameters(), 3)
      nan_check = False
      for param in model.parameters():
        if param.grad is not None:
          if torch.isnan(param.grad.sum()):
            nan_check = True
            break
      if not nan_check:
        optimizer.step()
        optimizer.zero_grad()

      eval_at -= 1
      if eval_at <= 0:
        with torch.no_grad():
          train_loss = np.mean(losses)
          train_gate = np.mean(gate)

          eval_at = args.evaluate_every
          losses = []
          gate = []

          tqdm.write("----------------------------------------------")
          tqdm.write("Epoch: %d" % (epoch))
          tqdm.write("Step: %d" % (step))
          tqdm.write('Train gate: %.4f' % train_gate)
          tqdm.write('Train loss: %.4f' % train_loss)
          if train_loss > 4:
            continue

          dev_acc = []
          dev_loss = []
          pred_sums = []
          gold_sums = []
          printing = 5
          for j in tqdm(range(0, len(x_dev), 1)):
            model.eval()

            x_batch = x_dev[j:j+1]
            y_batch = y_dev[j:j+1]

            x_batch = [[tokenizer.encode(x_rev) for x_rev in x_inst] for x_inst in x_batch]
            y_batch = [tokenizer.encode(y_inst) for y_inst in y_batch]

            tokens_batch, token_ids_batch, aspect_batch, sentiment_batch = utils.run_condense(x_batch, tokenizer, con_encoder, con_model)
            output_smooth_batch, output_mask_batch = utils.bert_label_smoothing(y_batch, tokenizer, language_model)

            tokens_batch = utils.pad_vector(tokens_batch, args.hidden_dim)[0]
            token_ids_batch, token_mask_batch = utils.pad_text(token_ids_batch)
            aspect_batch = torch.Tensor(aspect_batch).float().cuda() # batch size, hidden dim
            sentiment_batch = torch.Tensor(sentiment_batch).float().cuda() # batch size, hidden dim

            output_batch, _ = utils.pad_text(y_batch)

            pred_batch, _, loss = model(tokens_batch, token_ids_batch, token_mask_batch,
                                        aspect_batch, sentiment_batch, 
                                        output_batch, output_smooth_batch, output_mask_batch,
                                        dev=True)

            dev_acc.append(loss[1].item())

            output = output_batch[0].cpu().detach().numpy()
            pred = pred_batch[0].argmax(-1).cpu().detach().numpy()
            output = list(output)
            pred = list(pred)
            output = output[1:output.index(102)]
            try:
              pred = pred[:pred.index(102)]
            except:
              pass

            output = tokenizer.decode(output)
            pred = tokenizer.decode(pred)

            gold_sums.append(output)
            pred_sums.append(pred)
            if printing:
              printing -= 1
              tqdm.write('gold: %s' % output)
              tqdm.write('pred: %s' % pred)
              tqdm.write("----------------------------------------------")

          dev_acc = np.mean(dev_acc)
          tqdm.write('Dev ACC: %.4f' % dev_acc)

          if dev_acc >= best_acc:
            tqdm.write('UPDATING MODEL FILE...')
            best_acc = dev_acc
            stop_at = args.training_stopper
            torch.save({
              'model': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'dev_acc': dev_acc,
            }, abstract_file)
          else:
            stop_at -= 1
            tqdm.write("STOPPING AT: %d" % stop_at)

          tqdm.write("----------------------------------------------")


def evaluate(args):
  print(args)

  condense_file = 'model/%s/condense.model' % args.data_type
  abstract_file = 'model/%s/abstract.model' % args.data_type
  os.makedirs('output/%s/' % args.data_type, exist_ok=True)
  solution_file = 'output/%s/predictions.txt' % args.data_type

  tokenizer = BertTokenizer.from_pretrained(args.bert_config)
  tokenizer.add_special_tokens({'additional_special_tokens': ['<movie>']})
  vocab_size = len(tokenizer)

  print('Loading datasets...')
  x_test, y_test = utils.abstract_data(args.test_file, multi_ref=args.multi_ref)
  if args.data_type == 'rotten':
    m_test = utils.get_movies_from_file(args.test_file)

  print('Initializing models...')
  con_encoder = nn.Embedding(vocab_size, args.input_dim)
  con_encoder.requires_grad_(False)
  con_encoder.cuda()

  con_model = Condense(args.aspect_dim, args.sentiment_dim, args.input_dim, args.hidden_dim, vocab_size)
  con_model.requires_grad_(False)
  con_model.cuda()

  model = Abstract(vocab_size, args.hidden_dim, args.hidden_dim)
  model.requires_grad_(False)
  model.cuda()

  print('Loading models...')
  assert os.path.exists(condense_file)
  best_point = torch.load(condense_file)
  con_encoder.load_state_dict(best_point['encoder'])
  con_model.load_state_dict(best_point['model'])

  assert os.path.exists(abstract_file)
  best_point = torch.load(abstract_file)
  model.load_state_dict(best_point['model'])

  eval_at = args.evaluate_every
  stop_at = args.training_stopper

  f_sol = open(solution_file, 'w', encoding='utf-8', errors='ignore')
  printing = 5
  pred_sums = []
  print('Generating summaries...')
  for j in tqdm(range(0, len(x_test), 1)):
    model.eval()

    x_batch = x_test[j:j+1]
    y_batch = y_test[j:j+1]
    if args.data_type == 'rotten':
      m_batch = m_test[j:j+1]

    x_batch = [[tokenizer.encode(x_rev) for x_rev in x_inst] for x_inst in x_batch]
    y_batch = [tokenizer.encode(y_inst) for y_inst in y_batch]

    tokens_batch, token_ids_batch, aspect_batch, sentiment_batch = utils.run_condense(x_batch, tokenizer, con_encoder, con_model)

    tokens_batch = utils.pad_vector(tokens_batch, args.hidden_dim)[0]
    token_ids_batch, token_mask_batch = utils.pad_text(token_ids_batch)
    aspect_batch = torch.Tensor(aspect_batch).float().cuda() # batch size, hidden dim
    sentiment_batch = torch.Tensor(sentiment_batch).float().cuda() # batch size, hidden dim
    
    y_batch = [tokenizer.encode(y) for y in y_batch]
    output_batch, output_mask_batch = utils.pad_text(y_batch)

    pred_batch = model.beam_search(tokens_batch, token_ids_batch, token_mask_batch,
                                   aspect_batch, sentiment_batch, beam_size=args.beam_size, max_len=args.max_len)

    output = output_batch[0].cpu().detach().numpy()
    pred = pred_batch.cpu().detach().numpy()
    output = list([int(y) for y in output if int(y) != 101])
    pred = list([int(p) for p in pred if int(p) != 101])
    output = output[:output.index(102)]
    try:
      pred = pred[:pred.index(102)]
    except:
      pass
    output = tokenizer.decode(output)
    pred = tokenizer.decode(pred)
    if args.data_type == 'rotten':
      output = output.replace('<movie>', m_batch[0])
      pred = pred.replace('<movie>', m_batch[0])

    f_sol.write(pred + '\n')

    if printing:
      printing -= 1
      tqdm.write('gold: %s' % output)
      tqdm.write('pred: %s' % pred)
      tqdm.write("----------------------------------------------")

  f_sol.close()
  print('Summaries saved.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('-mode', default='train', type=str) # train or create

  parser.add_argument('-data_type', default='yelp', type=str) # rotten, yelp, or amazon

  parser.add_argument('-aspect_dim', default=100, type=int) # 50 (rotten), 100 (yelp or amazon)
  parser.add_argument('-sentiment_dim', default=5, type=int) # 2 (rotten), 5 (yelp or amazon)
  parser.add_argument('-adjust_sentiment', default=1, type=int) # 0 (rotten), 1 (yelp or amazon)
  parser.add_argument('-multi_ref', default=0, type=int) # 0 (rotten or yelp), 1 (amazon)

  parser.add_argument('-input_dim', default=256, type=int)
  parser.add_argument('-hidden_dim', default=256, type=int)

  parser.add_argument('-num_epoch', default=100, type=int)
  parser.add_argument('-batch_size', default=16, type=int)
  parser.add_argument('-learning_rate', default=3e-5, type=float)

  parser.add_argument('-warmup', default=8000, type=int)
  parser.add_argument('-evaluate_every', default=600, type=int)
  parser.add_argument('-training_stopper', default=20, type=int)

  parser.add_argument('-max_len', default=200, type=int)
  parser.add_argument('-beam_size', default=2, type=int)

  parser.add_argument('-train_file', default='data/yelp/train.plan.json', type=str)
  parser.add_argument('-dev_file', default='data/yelp/dev.json', type=str)
  parser.add_argument('-test_file', default='data/yelp/test.json', type=str)
  parser.add_argument('-bert_config', default='bert-base-uncased', type=str)

  args = parser.parse_args()
  if args.mode == 'train':
    train(args)
  elif args.mode == 'eval':
    evaluate(args)