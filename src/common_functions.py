
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.optim as optim


# class Conv_and_Pool(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, filter_width):
#         super(Conv_and_Pool, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.embedding_dim = embedding_dim
#         self.filter_width = filter_width
#
#         self.conv = nn.Conv2d(1, self.hidden_dim, (self.filter_width, self.embedding_dim))
#
#     def forward(self, x, mask):
#         pad_tensor3 = torch.zeros(x.size(0), self.conv.kernel_size[0]//2, x.size(2)).cuda()
#         x = torch.cat([pad_tensor3, x, pad_tensor3],1) #(batch, sent_len+width-1, emb_size)
#
#         x = x.unsqueeze(1) #(batch, 1, sent_len+width-1, emb_size)
#         x = F.relu(self.conv(x)).squeeze(3)  # (batch, #kernel, sent_len)
#
#         '''mask'''
#         x = x*mask.unsqueeze(1) #(batch, #kernel, sent_len)
#         mask_for_conv_output=(1.0-mask)*(mask-10.0)#(batch, sent_len)
#         x=x+mask_for_conv_output.unsqueeze(1)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2) #(batch, hidden)
#         return x



def conv_and_pool(x, mask, conv):
    '''
    x: (batch, sent_len, emb_size)
    mask: (batch, sent_len)
    conv(x): (batch, #kernal, sent_len-filter_width+1, 1)
    max_pool1d(x): (batch, #kernel, 1)
    '''
    pad_tensor3 = torch.zeros(x.size(0), conv.kernel_size[0]//2, x.size(2)).cuda()
    x = torch.cat([pad_tensor3, x, pad_tensor3],1) #(batch, sent_len+width-1, emb_size)

    x = x.unsqueeze(1) #(batch, 1, sent_len+width-1, emb_size)
    naked_conv_out = conv(x).squeeze(3)# (batch, #kernel, sent_len)
    x = F.relu(conv(x)).squeeze(3)  # (batch, #kernel, sent_len)

    '''mask'''
    x = x*mask.unsqueeze(1) #(batch, #kernel, sent_len)
    mask_for_conv_output=(1.0-mask)*(mask-10.0)#(batch, sent_len)
    x=x+mask_for_conv_output.unsqueeze(1)
    x = F.max_pool1d(x, x.size(2)).squeeze(2) #(batch, hidden)
    return x, naked_conv_out

def attentive_convolution(tensor3_l, tensor3_r, mask_l, mask_r, conv_wid_3, conv_wid_1):
    '''
    tensor3: (batch, sent_len, emb_size)
    mask: (batch, sent_len)
    '''
    tensor3_l = tensor3_l * mask_l.unsqueeze(2) #(batch, len_l, emb_size)
    tensor3_r = tensor3_r * mask_r.unsqueeze(2) #(batch, len_r, emb_size)
    inter_tensor3 = torch.bmm(tensor3_l, tensor3_r.permute(0,2,1)) #(batch, len_l, len_r)
    inter_tensor3_softmax = nn.Softmax(dim=2)(inter_tensor3)#(batch, len_l, len_r)
    weighted_tensor3_r = torch.bmm(inter_tensor3_softmax,tensor3_r)*mask_l.unsqueeze(2) #(batch, len_l, emb_size)



    _, conv_out_l = conv_and_pool(tensor3_l, mask_l, conv_wid_3)#(batch, #kernel, len_l)
    _, context_out_l = conv_and_pool(weighted_tensor3_r, mask_l, conv_wid_1)#(batch, #kernel, len_)

    '''combine'''
    combined_output_tensor3 = (F.relu(conv_out_l + context_out_l))*mask_l.unsqueeze(1) #(batch, #kernel, len_)
    mask_l_for_conv_output=(1.0-mask_l)*(mask_l-10.0) #(batch, len_l)
    before_maxpool_l = combined_output_tensor3 + mask_l_for_conv_output.unsqueeze(1)#(batch, #kernel, len_)
    max_pooling_output_l = F.max_pool1d(before_maxpool_l, before_maxpool_l.size(2)).squeeze(2) #(batch, #kernel)

    return max_pooling_output_l

def multi_channel_conv_and_pool(input_tensor3, mask, conv, conv2):
    '''
    x: (batch, sent_len, emb_size)
    mask: (batch, sent_len)
    conv(x): (batch, #kernal, sent_len-filter_width+1, 1)
    max_pool1d(x): (batch, #kernel, 1)
    '''
    x = input_tensor3
    pad_tensor3 = torch.zeros(x.size(0), conv.kernel_size[0]//2, x.size(2)).cuda()
    x = torch.cat([pad_tensor3, x, pad_tensor3],1) #(batch, sent_len+width-1, emb_size)

    x = x.unsqueeze(1) #(batch, 1, sent_len+width-1, emb_size)
    x = F.relu(conv(x)).squeeze(3)  # (batch, #kernel, sent_len)

    '''mask'''
    x = x*mask.unsqueeze(1) #(batch, #kernel, sent_len)
    mask_for_conv_output=(1.0-mask)*(mask-10.0)#(batch, sent_len)
    x=x+mask_for_conv_output.unsqueeze(1)
    output_1 = F.max_pool1d(x, x.size(2)).squeeze(2) #(batch, hidden)
    '''2nd conv'''
    x = input_tensor3
    pad_tensor3_2 = torch.zeros(x.size(0), conv2.kernel_size[0]//2, x.size(2)).cuda()
    x = torch.cat([pad_tensor3_2, x, pad_tensor3_2],1) #(batch, sent_len+width-1, emb_size)

    x = x.unsqueeze(1) #(batch, 1, sent_len+width-1, emb_size)
    x = F.relu(conv2(x)).squeeze(3)  # (batch, #kernel, sent_len)

    '''mask'''
    x = x*mask.unsqueeze(1) #(batch, #kernel, sent_len)
    mask_for_conv_output=(1.0-mask)*(mask-10.0)#(batch, sent_len)
    x=x+mask_for_conv_output.unsqueeze(1)
    output_2 = F.max_pool1d(x, x.size(2)).squeeze(2) #(batch, hidden)
    return torch.cat([output_1, output_2],1) #(batch, 2*hidden)

def LSTM(embeds, seq_lengths, lstm_model, batch_first):
    '''
    embeds: (batch, sent_len, emb_size)
    seq_lengths: a python list
    '''
    permuted_embeds = embeds.permute(1,0,2)
    packed_batch = pack_padded_sequence(permuted_embeds, seq_lengths, batch_first=batch_first)
    _, (last_hiddens, _)= lstm_model(packed_batch)
    return last_hiddens.view(embeds.size(0), last_hiddens.size(2))

def GRU(embeds, seq_lengths, gru_model, batch_first):
    '''
    embeds: (batch, sent_len, emb_size)
    seq_lengths: a python list
    '''
    permuted_embeds = embeds.permute(1,0,2)
    packed_batch = pack_padded_sequence(permuted_embeds, seq_lengths, batch_first=batch_first)
    _, last_hiddens= gru_model(packed_batch)
    return last_hiddens.view(embeds.size(0), last_hiddens.size(2))

def cosine_two_matrices(x1, x2, eps=1e-8):
    '''
    x1, x2: (batch, hidden)
    '''
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def torch_where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)

def normalize_matrix_rowwise_by_max(matrix):
    # print('before matrix:',matrix )
    matrix =  matrix/(1.0+torch.sum(matrix, dim=1, keepdim=True))
    # print('after matrix:',matrix )
    return matrix
