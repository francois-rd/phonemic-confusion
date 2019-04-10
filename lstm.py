"""
Basic LSTM Class with bi-directional option
Authors:                    Arnold Yeung
Upload Date:                April 6th, 2019
Revisions:
    2019-04-06      (AY)    Initial upload
    2019-04-09      (AY)    Added tensors2packedseq() / packedseq2tensors()
                            Added Embedding layer 
                            Modified create_sample_data(), train(), etc. to match data types
    
Helpful Links:
    PyTorch LSTM outputs : https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
    Example of PackedSequence : https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class lstm_rnn(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        
        self.input_dim = params['input_dim']
        self.embed_dim = params['embed_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.num_layers = params['num_layers']
        self.bidirectional = params['bidirectional']
        
        self.batch_size = params['batch_size']
        self.hidden_states = self.init_hidden_states(self.bidirectional)
        self.cell_state = self.init_cell_state(self.bidirectional)
        
        #   Embedding layer - size vocabSize x vocabSize
        self.embedding = nn.Embedding(self.input_dim, self.embed_dim)
        
        #   LSTM hidden layer
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers, 
                            bidirectional=self.bidirectional)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
    
    def init_hidden_states(self, bidirection=False):
        """
        Initialize hidden states to random.
        Arguments:
            bidirection (boolean) : whether LSTM is bidirectional or not
        Returns:
            hidden state (torch.tensor) : hidden states per layer (1 state per layer)
        """
        if bidirection is True:
             return torch.randn(self.num_layers*2, self.batch_size, self.hidden_dim)
        else:
            return torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
    
    def init_cell_state(self, bidirection=False):
        """
        Initialize cell states to 0.
        Arguments:
            bidirection (boolean) : whether LSTM is bidirectional or not
        Returns:
            cell state (torch.array) : cell states per layer (1 state per layer)
        """
        if bidirection is True:
             return torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim)
        else:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        
    def forward(self, X, seq_lens):
        """
        Forward pass of model
        Arguments:
            input:    input matrix of size(seq_length, batch_size, input_dim)
        Return:
            y_pred:   predicted output of each sequence step
        """
        
        #   reset the LSTM network
        self.hidden_states = self.init_hidden_states(self.bidirectional)
        self.cell_state = self.init_cell_state(self.bidirectional)
        
        #   X input shape (batch_size, max(seq_lens))
        #   output shape (batch_size x max(seq_lens) x input_dim)
        X = self.embedding(X)

        #X = X.transpose(dim0=0, dim1=1)     #   shape (seq_lens x batch_size, input_dim)
        X = self.tensors2packedseq(X, seq_lens)
    
        #   lstm_out is output for each t in sequence length T
        #   final_hidden_states are hidden_states for t=T
        #   final_cell_state is cell_state for t=T
        
        lstm_out, (final_hidden_states, final_cell_state) = self.lstm(X, 
                  (self.hidden_states, self.cell_state))
                
        #   convert lstm_out back to tensor from packedsequence
        lstm_out, seq_lens = self.packedseq2tensors(lstm_out)
        
        # NOTE: final_hidden_states[-1,0,:] is same as lstm_out[-1,0,:] 
        #           shape (seq_len x batch_size x input_dim)
        #       BUT final_hidden_states[-1,a,:] is NOT same as lstm_out[-1, a, :] if 
        #           seq_lens[0] != seq_lens[a]
        #           instead final_hidden_states[-1,a,:] = lstm_out[-1*(seq_lens[0] 
        #               - seq_lens[1] + 1), a, :])]
        #           That is, lstm_out will include "NULL" outputs in seq_len where
        #               there is no input, in order to keep a square tensor
        #           final_hidden_states[-1] however will only output UP TO the seq_len
        #               where there is input (i.e. ignore all NULL outputs)
        
        y_pred = torch.sigmoid(self.linear(final_hidden_states[-1]))
        
        return y_pred
        
    def compute_loss(self, y, target):
        """
        Compute the loss function using the MSE Loss function
        Arguments:
            y (torch.tensor) : outputs of classifier for input (batch_size, output_dim)
            target (torch.tensor) : ground truth labels for input (batch_size, output_dim)
        Returns:
            loss (torch.tensor) : MSE loss of y
        """
        return F.mse_loss(y, target)


    def tensors2packedseq(self, tensors, seq_lens):
        """
        Converts a 3d padded tensor (with ORIGINALLY variable sequence lengths) to a PackedSequence
        datatype)
        Arguments:
            tensors (torch.tensor) : padded input tensor of shape (batch_size, max_seq_len, input_dim)
            seq_lens (list[int]) : unpadded sequence length of each tensor in tensors[a, :, :]
        Returns:
            packedseq (PackedSequence) : a PackedSequence type of batch of input tensors 
        """
        
        #   convert 3d tensor to list of 2d tensors
        seq_lens_idx = np.flip(np.argsort(seq_lens), axis=0).tolist()     # descending indices based on seq_lens
        
        tensor_list = []
        seq_lens_list = []
        for idx in seq_lens_idx:                    #   for every sample in batch
            tensor_list.append(tensors[idx, :, :])
            seq_lens_list.append(seq_lens[idx])
                
        packedseq = nn.utils.rnn.pad_sequence(tensor_list)  #   tensor (seq_len, batch_size, input_dim)
        packedseq = nn.utils.rnn.pack_padded_sequence(packedseq, seq_lens_list)
    
        return packedseq
    
    def packedseq2tensors(self, packedseq):
        """
        Converts a PackedSequence to a tuple containing tensors of variable lengths 
        datatype)
        Arguments:
            packedseq (PackedSequence) : a PackedSequence type of batch of input tensors 
        Returns:
            tensors (torch.tensor) : padded input tensors of shape (seq_len, batch_size, input_dim)
            seq_lens (torch.tensor) : tensor containing the seq_len of each input tensor (seq_len, )
        """       
        tensors, seq_lens = nn.utils.rnn.pad_packed_sequence(packedseq)
        tensors = tensors.contiguous()
        
        return tensors, seq_lens

def pad_lists(lists, pad_token=0):
    """
    Pads lists of different lengths to all have the same length (max length)
    Arguments:
        lists : list of 1d lists with different lengths (list[list[int]])
        pad_token : padding value (int)
    Returns:
        lists (list[list[int]]) : List of padded 1d lists with equal lengths 
        seq_lens (list[int]) : List of sequence lengths of corresponding lists (i.e. len(lst))
    """
    seq_lens = [len(lst) for lst in lists]
    max_seq_len = max(seq_lens)
    lists = [lst + [pad_token]*(max_seq_len - len(lst)) for lst in lists]
    
    return lists, seq_lens


def train(X_train, y_train, clf, params):
    """
    Trains the model given training data.
    Arguments:
        X_train (torch.tensor) : padded tensor containing integers (1d, instead of hidden_dim) 
            of shape (seq_len x num_sample)
        y_train (torch.tensor()) : expected output labels corresponding to X
        clf (class) : lstm model
        params (dict) : contains model, dimension, and batch parameters
    Returns:
        None
    """
    
    num_epochs = params['num_epochs']
    lr = params['learning_rate']
    #   Create optimizer
    
    optimizer = torch.optim.Adam(list(clf.parameters()), lr=lr)

    for epoch in range(0, num_epochs):
        
        #   get batch
        X_train_batch, X_train_seq_lens = pad_lists(X_train)
        X_train_batch = torch.tensor(X_train_batch)

        #   Zero the gradient of the optimizer
        optimizer.zero_grad()
        
        #   Forward pass
        y_pred = clf.forward(X_train_batch, X_train_seq_lens)
        loss = clf.compute_loss(y_pred, y_train)
        
        print("Epoch: " + str(epoch) + "    Loss: " + str(loss.item()))
    
        loss.backward()     #   backward pass
        optimizer.step()    #   update gradient step

def evaluate(X_test, y_test, clf, params):
    
    X_test, X_test_seq_lens = pad_lists(X_test)
    X_test = torch.tensor(X_test)
    
    y_pred = clf.forward(X_test, X_test_seq_lens)
    loss = clf.compute_loss(y_pred, y_test)
    
    return loss, y_pred

def create_sample_data(params, min_seq_len, max_seq_len):
    """
    **Don't use this if you don't know it.  Buggy and only for verifying code runs.**
    
    Creates data X, y for a very simple test case.  For first half of the batch
    will be labelled 1 and will have X < vocabSize/2.  The second half of the batch will
    be labelled 0 and will have X typically > vocabSize/2.
    Arguments:
        params (dict) : contains dimension and batch parameters
        min_seq_len (int) : length of the shortest possible sequence
        max_seq_len (int) : length of the longest possible sequence
    Returns:
        List of integer sequences in X (List[list[int]])
        torch.tensor(batch_size, output_dim) : y labels corresponding to X
    """
    
    vocabSize = params['embed_dim']
    output_dim = params['output_dim']
    batch_size = params['batch_size']
           
    X = []
    
    #   generate random sorted seq_len
    seq_len = torch.randint(min_seq_len, max_seq_len, (batch_size,))       
    seq_len = seq_len.sort(descending=True)[0]
    
    #   create y
    y = torch.ones((batch_size,output_dim))
    y[int(batch_size/2):, ] = 0
    
    #   create data sequences of different sequence lengths
    for i in range(0, batch_size):
        if i > int(batch_size/2)-1:
            sequence = [random.randrange(1, int(vocabSize/2)) for _ in range(seq_len[i])]
        else:
            sequence = [random.randrange(int(vocabSize/2), vocabSize) for _ in range(seq_len[i])]
    
        X.append(sequence)

    return X, y

def main_fcn(params, verbose=False):
    
    min_seq_len = 1
    max_seq_len = 20
    
    #   initialize classifier
    if verbose:
        print("Initializing classifier...")
        
    #   Create model
    clf = lstm_rnn(params)
    
    #   Train model
    X_train, y_train = create_sample_data(params, min_seq_len, max_seq_len)
    X_test, y_test = create_sample_data(params, min_seq_len, max_seq_len)
    
    train(X_train, y_train, clf, params)
    
    #   Evaluate model
    loss, output = evaluate(X_test, y_test, clf, params)
    print(loss)
    print(output)
    
if __name__ == '__main__':
    
    torch.manual_seed = 1
    vocab_size = 26
    
    params = {
            'input_dim':        vocab_size,
            'embed_dim':        vocab_size,
            'hidden_dim':       20,            
            'output_dim':       1,              
            'num_layers':       5,
            'batch_size':       50,
            'num_epochs':       200,
            'learning_rate':    0.05,
            'learning_decay':   0.9,
            'bidirectional':    True
    }
    
    main_fcn(params, verbose=True)
