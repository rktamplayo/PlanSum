import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math
import utils


class GradientReverse(torch.autograd.Function):

  lambd = 1.0

  @staticmethod
  def forward(ctx, inp):
    out = inp.clone()
    return out

  @staticmethod
  def backward(ctx, grad_out):
    return grad_out.neg() * GradientReverse.lambd


def reverse_gradient(x, lambd):
  GradientReverse.lambd = lambd
  return GradientReverse.apply(x)


class Condense(nn.Module):

  def __init__(self, aspect_dim, sentiment_dim, input_dim, hidden_dim, vocab_size):
    super(Condense, self).__init__()
    self.aspect_dim = aspect_dim
    self.sentiment_dim = sentiment_dim
    self.vocab_size = vocab_size
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    self.aspect_embedding = nn.Parameter(torch.Tensor(aspect_dim, hidden_dim//2))
    nn.init.kaiming_normal_(self.aspect_embedding)
    self.sentiment_embedding = nn.Parameter(torch.Tensor(sentiment_dim, hidden_dim//2))
    nn.init.kaiming_normal_(self.sentiment_embedding)

    self.encoder = nn.LSTM(input_dim, hidden_dim//2, bidirectional=True, batch_first=True)
     
    self.doc_asp_classifier = nn.Linear(hidden_dim//2, aspect_dim)
    self.doc_sen_classifier = nn.Linear(hidden_dim//2, sentiment_dim)

    self.adv_classifier = nn.Linear(hidden_dim//2, sentiment_dim)

    self.dropout = nn.Dropout(0.5)


  def forward(self, tokens, mask, outputs, lambd=1):
    tokens = self.encoder(tokens)
    tokens = tokens[0]

    doc = torch.sum(tokens*mask.unsqueeze(-1), dim=1) # before encoding

    a_doc, s_doc = doc.chunk(2, -1)

    prob_a = F.softmax(self.doc_asp_classifier(a_doc), dim=1)
    prob_s = F.softmax(self.doc_sen_classifier(s_doc), dim=1)
    
    aspect = torch.matmul(prob_a, self.aspect_embedding)
    sentiment = torch.matmul(prob_s, self.sentiment_embedding)
    
    # ADVERSARIAL
    adv_a_doc = reverse_gradient(a_doc, lambd)
    adv_prob_s = F.softmax(self.adv_classifier(adv_a_doc), dim=1)

    return (a_doc, s_doc), (aspect, sentiment), prob_s, adv_prob_s


  def condense(self, tokens, mask):
    tokens = self.encoder(tokens)
    tokens = tokens[0]

    doc = torch.sum(tokens*mask.unsqueeze(-1), dim=1) # before encoding
    a_doc, s_doc = doc.chunk(2, -1)
 
    prob_a = F.softmax(self.doc_asp_classifier(a_doc), dim=1)
    prob_s = F.softmax(self.doc_sen_classifier(s_doc), dim=1)

    return tokens, doc, prob_a, prob_s


  def calculate_loss(self, before, after, sen_pred, adv_pred, sen_gold):
    before = before[0].view(-1, self.hidden_dim//2)
    after = after[0].view(-1, self.hidden_dim//2)

    asp_loss = torch.zeros(1).cuda()
    pos_sim = (before*after).sum(1)
    for k in range(5):
      shuffle_indices = np.random.permutation(np.arange(before.size()[0]))
      negative = before[shuffle_indices]
      neg_sim = (negative*after).sum(1) 
      asp_loss = torch.max(asp_loss, 1-pos_sim+neg_sim)
    asp_loss = asp_loss.mean()

    asp_norm = torch.norm(self.aspect_embedding, dim=-1, keepdim=True)
    asp_norm = self.aspect_embedding / asp_norm
    asp_norm_loss = torch.matmul(asp_norm, asp_norm.t()) - torch.eye(self.aspect_dim).cuda()
    asp_norm_loss = asp_norm_loss.abs().sum()

    sen_pred = sen_pred + 1e-9
    sen_pred = sen_pred / sen_pred.sum(dim=-1, keepdim=True)
    sen_loss = F.nll_loss(torch.log(sen_pred), sen_gold)

    sen_norm = torch.norm(self.sentiment_embedding, dim=-1, keepdim=True)
    sen_norm = self.sentiment_embedding / sen_norm
    sen_norm_loss = torch.matmul(sen_norm, sen_norm.t()) - torch.eye(self.sentiment_dim).cuda()
    sen_norm_loss = sen_norm_loss.abs().sum()

    adv_pred = adv_pred + 1e-9
    adv_pred = adv_pred / adv_pred.sum(dim=-1, keepdim=True)
    adv_loss = F.nll_loss(torch.log(adv_pred), sen_gold)

    return asp_loss, asp_norm_loss, sen_loss, sen_norm_loss, adv_loss


  def get_aspect(self, prob_a):
    return torch.matmul(prob_a, self.aspect_embedding)


  def get_sentiment(self, prob_s):
    return torch.matmul(prob_s, self.sentiment_embedding)


class Abstract(nn.Module):

  def __init__(self, vocab_size, input_dim, hidden_dim):
    super(Abstract, self).__init__()
    self.vocab_size = vocab_size
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    self.embedding = nn.Embedding(vocab_size, input_dim)
    self.iso_transform = nn.Linear(input_dim, hidden_dim)
    self.iso_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim))

    self.ht_transform = nn.Linear(hidden_dim, 2*hidden_dim)
    self.yt_transform = nn.Linear(input_dim+hidden_dim, hidden_dim)
    
    self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    self.attend_key = nn.Linear(hidden_dim, hidden_dim)
    self.attend_query = nn.Linear(hidden_dim, hidden_dim)
    self.attend_weight = nn.Linear(hidden_dim, 1)

    self.pointer = nn.Linear(hidden_dim*3, 1)

    self.att_classifier = nn.Linear(hidden_dim, vocab_size)

    self.dec_classifier = nn.Linear(hidden_dim, vocab_size)
    self.dropout = nn.Dropout(0.5)


  def forward(self, tokens, token_ids, token_mask,
              aspect, sentiment, 
              output, output_smooth, output_mask,
              dev=False):
    batch_size, token_len, hidden_dim = tokens.size()
    _, output_len = output.size()
    output_len -= 1

    xt = self.iso_transform(self.embedding(token_ids))
    xt = xt + tokens
    tokens = self.iso_mlp(xt)

    zt = torch.cat([aspect, sentiment], dim=-1)

    # embed output tokens
    yt = self.embedding(output[:,:-1])
    yt = self.dropout(yt)
    zt = zt.unsqueeze(1).expand(-1, output_len, -1)
    
    yzt = self.yt_transform(torch.cat([yt, zt], dim=-1))

    # decode, attend, point
    input_ = (tokens*token_mask.unsqueeze(-1)).sum(dim=1)
    input_ = input_ / token_mask.mean(dim=1).unsqueeze(-1)
    st, ct = self.ht_transform(input_).chunk(2, dim=-1)
    st = st.unsqueeze(0).contiguous()
    ct = ct.unsqueeze(0).contiguous()

    st, _ = self.decoder(yzt, (st, ct))

    kt = self.attend_key(tokens).unsqueeze(1) # batch size, 1, token len, hidden dim
    qt = self.attend_query(st).unsqueeze(2) # batch size, output len, 1, hidden dim
    at = self.attend_weight((kt+qt).tanh()).softmax(dim=2) # batch size, output len, token len, 1
    at = at * token_mask.unsqueeze(1).unsqueeze(-1)
    at = at / at.sum(dim=2, keepdim=True)
    vt = (tokens.unsqueeze(1)*at).sum(dim=2) # batch size, output len, hidden dim
    at = at.squeeze(-1)

    gt = self.pointer(torch.cat([yzt,st,vt], dim=-1)).sigmoid()
    p_copy = torch.zeros(batch_size, output_len, self.vocab_size).cuda()
    bindex = torch.arange(0, batch_size).unsqueeze(-1).expand(-1, token_len*output_len).contiguous().view(-1)
    oindex = torch.arange(0, output_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, token_len).contiguous().view(-1)
    tindex = torch.arange(0, token_len).unsqueeze(0).unsqueeze(0).expand(batch_size, output_len, -1).contiguous().view(-1)
    vindex = token_ids.unsqueeze(1).expand(-1, output_len, -1).contiguous().view(-1)
    p_copy[bindex,oindex,vindex] += at[bindex,oindex,tindex]

    p_generate = (self.dec_classifier(st) + self.att_classifier(vt)).softmax(dim=-1)
    pt = gt * p_generate + (1-gt) * p_copy

    mask = output_mask[:,1:].contiguous()
    log_pt = torch.log(pt+1e-9) # b, s, v
    yt_ = output_smooth[:,1:].contiguous() # b, s, v

    loss = -log_pt * yt_ # b, s, v
    loss = loss.sum(-1) * mask # b, s
    loss = loss.sum(-1) / mask.sum(-1) # b
    loss = loss.mean()
    if dev:
      loss2 = pt.argmax(-1).eq(output[:,1:]) * mask # b, s
      loss2 = loss2.sum(1) / mask.sum(1) # b
      loss2 = loss2.mean()

      loss = (loss, loss2)

    return pt, gt, loss


  def beam_search(self, tokens, token_ids, token_mask,
                  aspect, sentiment, start_idx=101, end_idx=102, 
                  beam_size=1, max_len=200, dev=False):
    batch_size, token_len, hidden_dim = tokens.size()

    xt = self.iso_transform(self.embedding(token_ids))
    xt = xt + tokens
    tokens = self.iso_mlp(xt)

    zt = torch.cat([aspect, sentiment], dim=-1)

    input_ = (tokens*token_mask.unsqueeze(-1)).sum(dim=1)
    input_ = input_ / token_mask.mean(dim=1).unsqueeze(-1)
    s0, c0 = self.ht_transform(input_).chunk(2, dim=-1)
    zt = zt.unsqueeze(1)

    s0 = s0.view(1, 1, self.hidden_dim)
    c0 = c0.view(1, 1, self.hidden_dim)

    beam = [{
      'input': [start_idx], #torch.tensor([start_idx]).cuda(),
      'prob': 0,
      'prob_norm': 0,
      'trigrams': []
    }]
    finished = []

    while len(beam) != 0:
      new_beam = []

      inp_batch = [instance['input'] for instance in beam]
      inp_batch, _ = utils.pad_text(inp_batch)

      yt = self.embedding(inp_batch)

      batch_size, output_len, _ = yt.size()
      yzt = self.yt_transform(torch.cat([yt, zt.expand(batch_size, output_len, -1)], dim=-1))

      s0_ = s0.expand(-1, batch_size, -1).contiguous()
      c0_ = c0.expand(-1, batch_size, -1).contiguous()

      st, _ = self.decoder(yzt, (s0_, c0_))

      kt = self.attend_key(tokens).unsqueeze(1)
      qt = self.attend_query(st).unsqueeze(2) # batch size, output len, 1, hidden dim
      at = self.attend_weight((kt+qt).tanh()).softmax(dim=2) # batch size, output len, token len, 1
      at = at * token_mask.unsqueeze(1).unsqueeze(-1)
      at = at / at.sum(dim=2, keepdim=True)
      vt = (tokens.unsqueeze(1)*at).sum(dim=2) # batch size, output len, hidden dim
      at = at.squeeze(-1)

      gt = self.pointer(torch.cat([yzt,st,vt], dim=-1)).sigmoid()
      p_copy = torch.zeros(batch_size, output_len, self.vocab_size).cuda()
      bindex = torch.arange(0, batch_size).unsqueeze(-1).expand(-1, token_len*output_len).contiguous().view(-1)
      oindex = torch.arange(0, output_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, token_len).contiguous().view(-1)
      tindex = torch.arange(0, token_len).unsqueeze(0).unsqueeze(0).expand(batch_size, output_len, -1).contiguous().view(-1)
      vindex = token_ids.unsqueeze(1).expand(batch_size, output_len, -1).contiguous().view(-1)
      p_copy[bindex,oindex,vindex] += at[bindex,oindex,tindex]

      p_generate = (self.dec_classifier(st) + self.att_classifier(vt)).softmax(dim=-1)
      pt_batch = gt * p_generate + (1-gt) * p_copy

      for pt, instance in zip(pt_batch, beam):
        inp = instance['input']
        prob = instance['prob']
        trigrams = instance['trigrams']

        if len(inp) == max_len:
          finished.append(instance)
          continue
        if inp[-1] == end_idx:
          finished.append(instance)
          continue

        pt = pt[len(inp)-1]

        pk, yk = torch.topk(pt, k=20, dim=-1)
        count = 0
        nuclear = 0
        for pt, yt in zip(pk, yk):
          if count == beam_size:
            break

          if not dev:
            if yt == end_idx and len(inp) < 10:
              continue
            if len(inp) >= 1:
              if inp[-1] == yt:
                continue
            if len(inp) >= 1:
              if tuple(inp[-1:] + [yt.item()]) in trigrams:
                continue
            if len(inp) >= 3:
              if inp[-3:-1] == inp[-1:] + [yt.item()]:
                continue

          count += 1
          new_instance = {
            'input': inp + [yt.item()], #torch.cat([inp, yt.unsqueeze(0)], dim=-1),
            'prob': prob + torch.log(pt),
            'prob_norm': (prob + torch.log(pt)) / (len(inp) + 1),
            'prob_ln': (prob + torch.log(pt)) / ((5 + len(inp)) ** 0.6 / 6 ** 0.6),
            'trigrams': trigrams + [tuple(inp[-2:])]
          }
          new_beam.append(new_instance)

      beam = sorted(new_beam, key=lambda a: -a['prob_norm'])[:beam_size]

    finished = sorted(finished, key=lambda a: -a['prob_norm'])[0]
    return torch.Tensor(finished['input']).cuda()
