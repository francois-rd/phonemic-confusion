"""
Transformer in PyTorch
Authors:                Arnold Yeung
Upload Date:            April 18th, 2019
Revisions:
    
    THIS CODE IS UNFINISHED AND ABANDONED.
    
Helpful Links:
    How to code the transformer in Pytorch : https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
    
Code based on the following sources:
    How to code the transformer in Pytorch : https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
    University of Toronto CSC421 Programming Assignment 4 : https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/
     
"""

MAX_SEQ_LEN = 100
SMALL_NUM = 1e-6

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from lstm_rnn import train, create_sample_data

class PositionalEncoder(nn.Module):
    
    def __init__(self, params, max_seq_len):
        super().__init__()
        
        self.embed_dim = params['embed_dim']
        self.batch_size = params['batch_size']
        max_seq_len = MAX_SEQ_LEN

        self.PE_constant = torch.zeros(max_seq_len, self.embed_dim)
        
        for row in range(0, max_seq_len):
            for col in range(0, self.input_dim, 2):
                self.PE_constant[row, col] = math.sin(row / (10000 ** (2*col / self.embed_dim)))
                self.PE_constant[row, col+1] = math.col(row / (10000 ** (2*(col+1) / self.embed_dim)))
        
        #   copy batch_size times
        #   shape : max_seq_len x batch_size x embed_dim
        self.PE_constant.unsqueeze(1)       
        self.PE_constant.expand(max_seq_len, self.batch_size, self.embed_dim)
    
    def forward(self, X):
        """
        Arguments:
            X (torch.tensor) : 3d input tensor of size (max(seq_lens) x batch_size, embed_dim)
        Returns:
            Same as above with added Positioning Encoding constants
        """
        #   increase embedding value magnitude to prevent loss of values
        X = X * math.sqrt(self.embed_dim)
        seq_len = X.shape[0]
        X = X + self.PE_constant[:seq_len,:,:]
        return X

class MultiHeadAttention(nn.Module):
    
    def __init__(self, params, dropout=0.25):
        super().__init__()
        
        self.embed_dim = params['embed_dim']
        
        self.q_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.embed_dim, self.embed_dim)
    
    def calculate_attention(self, query, value, key, scaling_factor, mask=True, dropout=True):
        
        numerator = torch.matmul(query, key.transpose(dim0=1, dim1=2)) * scaling_factor
        
        if mask is True:
            mask = torch.ones(1,1).expand_as(numerator).tril(diagonal=-1).cuda()
            numerator = numerator.triu() + mask * self.neg_inf
        
        attention_weights = F.softmax(numerator)
        
        if dropout is True:
            attention_weights = self.dropout(attention_weights)
    
        context = torch.matmul(attention_weights.transpose(dim0=1, dim1=2), value)
    
        return context, attention_weights
        

    def forward(self, query, value, key):
        
        query = self.q_linear(query)
        value = self.v_linear(value)
        key = self.k_linear(key)
        scaling_factor = torch.rsqrt(torch.tensor(self.embed_dim, dtype= torch.float))
        
        context, _ = calculate_attention(query, value, key, scaling_factor)

class AddNorm(nn.Module):
    
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, context, new_context):
    
        residual_context = context + new_context        #    concatenate contexts

        #   normalize    
        norm = (residual_context - residual_context.mean(dim=-1, keepdim=True))
        norm = norm /(residual_context.std(dim=-1, keepdim=True) + SMALL_NUM)
        
        return norm

class Encoder(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        
        #   initial layers
        self.input_dim = params['input_dim']
        self.embed_dim = params['embed_dim']
        self.ff_dim = params['ff_dim']
        self.max_seq_len = params['max_seq_len']
        self.num_layers = params['num_enc_layers']          # should be 1
        
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.positioning = PositionalEncoder(params, self.max_seq_len)
        self.addnorm = AddNorm()
        
        for layer in range(0, self.num_layers):
           self.multihead_attention[layer] = MultiHeadAttention(params, self.dropout)
           self.feedforward[layer] = nn.Sequential(F.relu(nn.Linear(self.embed_dim, self.ff_dim)),
                                         nn.Linear(self.ff_dim, self.embed_dim))
    
    def forward(self, X):
        
        embedded_X = self.embedding(X)
        context = self.positioning(embedded_X)      #   input to encoder layers
        
        for layer in range(0, num_layers):      #   repeat this layer multiple times
            new_context, new_attention_weights = self.multihead_attention[layer](context, 
                                                                 context, context)
            residual_context =  self.addnorm(context, new_context)         #   concatenate context
            new_context = self.feedforward[layer](residual_context)
            residual_context = self.addnorm(residual_context, new_context)
        
            context = residual_context
        
        return context, new_attention_weights
        
        

class Decoder(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        
        self.input_dim = params['input_dim']
        self.embed_dim = params['embed_dim']
        self.output_dim = params['output_dim']
        self.batch_size = params['batch_size']
        self.max_seq_len = params['max_seq_len']
        self.num_layers = params['num_dec_layers']
        self.dropout = params['dropout']
        
        #   initial layers
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.positioning = PositionalEncoder(params, self.max_seq_len)
        self.addnorm = AddNorm()
        self.output_linear = nn.Linear(self.embed_dim, self.output_dim)
        
        for layer in range(0, num_layers):      #   repeat this layer multiple times
            self.masked_multihead_attention[layer] = MultiHeadAttention(params, self.dropout, mask=True)
            self.encoder[layer] = Encoder(params)
            self.multihead_attention[layer] = MultiHeadAttention(params, self.dropout)
            self.feedforward[layer] = nn.Sequential(F.relu(nn.Linear(self.embed_dim, self.ff_dim)),
                                         nn.Linear(self.ff_dim, self.embed_dim))
        
    def forward(self, X):

        #   input into decoder should be target
        #   input into encoder should be source
        
        embedded_X = self.embedding(X)
        context = self.positioning(embedded_X)      #   input to decoder layers
        
        for layer in range(0, self.num_layers):
            
            #   1st multihead attention layer
            new_context, dec_attention_weights = self.masked_multihead_attention[layer](context, 
                                                                 context, context)
            residual_context =  self.addnorm(context, new_context)         #   concatenate context
            
            #   encoder layer (should we just past the output into here?)
            enc_context, enc_attention_weights = self.encoder[layer]()
            
            #   2nd multihead attention layer
            new_context, dec_attention_weights = self.multihead_attention[layer](residual_context,
                                                                         enc_context, enc_context)
            residual_context = self.addnorm(residual_context, new_context)
            
            #   feedforward layer
            new_context = self.feedforward[layer](residual_context)
            residual_context = self.addnorm(residual_context, new_context)
            
            context = residual_context
            
        return context
            
class transformer(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        
        self.input_dim = params['input_dim']
        self.embed_dim = params['embed_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.num_layers = params['num_layers']
        self.scalar = params['scalar']
        self.batch_size = params['batch_size']
        self.max_seq_len = params['max_seq_len']
        
        #   layers
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.positioning = PositionalEncoder(params, self.max_seq_len)
    
    def forward(self, X):
        
    
        output = F.softmax(self.output_linear(context), dim=-1)
        
        return output
    
    def compute_loss(self, y, target):
    

def main():
    pass

if __name__ == "__main__":
    pass