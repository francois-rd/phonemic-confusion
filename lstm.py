"""
Basic LSTM Class with bi-directional option
Authors:                    Arnold Yeung
Upload Date:                April 6th, 2019
Revisions:
    2019-04-06      (AY)    Initial upload
    2019-04-09      (AY)    Added tensors2packedseq() / packedseq2tensors()
                            Added Embedding layer 
                            Modified create_sample_data(), train(), etc. to match data types
                            Fixed shuffling of X in tensor2packedseq(), changed ordering to pad_lists()
                            Added ordering for y to pad_lists()
                            Added X_test, y_test output for evaluate()
    2019-04-10      (AY)    Added in non-scalar experiments (params['scalar']) and support functions for 1-hot
                            Modified existing functions for both params['scalar'] = True and False
                            Fixed bug in bidirectional scalar - now concatenates leftmost reverse hidden state 
                                and rightmost forward hidden state
    2019-04-11      (AY)    Add in DataLoader
                            Change list2onehottensors() / onehottensors2classlist() to have faster method using sklearn
    2019-04-12      (AY)    Removed warnings
                            Modified for new Dataloader
                            Made GPU compatible (cuda.()) for 'full'
                            Add basic training model saving feature
    2019-04-13      (AY)    Added scaling to make 'scalar' compute the # of errors (instead of word error rate)
                            Added epoch loss saving to report
                            Change scalar to RELU activation
                            Added in best model saving based on validation loss
                            
                            

Helpful Links:
    PyTorch LSTM outputs : https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
    Example of PackedSequence : https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
    Bidirectional LSTM Output : https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
    Bidirectional LSTM Output h_n : https://discuss.pytorch.org/t/how-can-i-know-which-part-of-h-n-of-bidirectional-rnn-is-for-backward-process/3883
    One Hot Encoding (Sklearn) : https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    Saving PyTorch models : https://medium.com/udacity-pytorch-challengers/saving-loading-your-model-in-pytorch-741b80daf3c
    
"""
import os, sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import src.phonemes
from src.data_loader import DataLoader
from sklearn.preprocessing import OneHotEncoder
import datetime as dt
from src.utils import int2phoneme

#   hide warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

DATA_DIR = "./data/transcripts/"
PHONEME_OUT = "./data/phonemes.txt"
SAVE_DIR = "./output/NN/LSTM/"

class lstm_rnn(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        
        self.input_dim = params['input_dim']
        self.embed_dim = params['embed_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.num_layers = params['num_layers']
        self.bidirectional = params['bidirectional']
        self.scalar = params['scalar']
        self.batch_size = params['batch_size']
        
        #self.hidden_states = self.init_hidden_states(self.bidirectional).cuda()
        #self.cell_state = self.init_cell_state(self.bidirectional).cuda()
        
        #   Embedding layer - size vocabSize x vocabSize
        self.embedding = nn.Embedding(self.input_dim, self.embed_dim)
        
        #   LSTM hidden layer
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers, 
                            bidirectional=self.bidirectional)
        
        if self.bidirectional is True:
            self.linear = nn.Linear(self.hidden_dim*2, self.output_dim)
        else:
            self.linear = nn.Linear(self.hidden_dim, self.output_dim)
    
    def init_hidden_states(self, batch_size, bidirection=False):
        """
        Initialize hidden states to random.
        Arguments:
            bidirection (boolean) : whether LSTM is bidirectional or not
        Returns:
            hidden state (torch.tensor) : hidden states per layer (1 state per layer)
        """
        if bidirection is True:
             return torch.randn(self.num_layers*2, batch_size, self.hidden_dim)
        else:
             return torch.randn(self.num_layers, batch_size, self.hidden_dim)
    
    def init_cell_state(self, batch_size, bidirection=False):
        """
        Initialize cell states to 0.
        Arguments:
            bidirection (boolean) : whether LSTM is bidirectional or not
        Returns:
            cell state (torch.array) : cell states per layer (1 state per layer)
        """
        if bidirection is True:
             return torch.zeros(self.num_layers*2, batch_size, self.hidden_dim)
        else:
             return torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        
    def forward(self, X, seq_lens):
        """
        Forward pass of model
        Arguments:
            input:    input matrix of size(seq_length, batch_size, input_dim)
        Return:
            y_pred:   predicted output of each sequence step
        """
        
        batch_size = X.shape[0]
        
        #   reset the LSTM network
        self.hidden_states = self.init_hidden_states(batch_size, self.bidirectional).cuda()
        self.cell_state = self.init_cell_state(batch_size, self.bidirectional).cuda()
        
        #   X input shape (batch_size, max(seq_lens))
        #   output shape (batch_size x max(seq_lens) x input_dim)
        X = self.embedding(X.cuda())

        X = self.tensors2packedseq(X, seq_lens)
    
        #   lstm_out is output for each t in sequence length T
        #   final_hidden_states are hidden_states for t=T
        #   final_cell_state is cell_state for t=T

        lstm_out, (final_hidden_states, final_cell_state) = self.lstm(X, 
                  (self.hidden_states.cuda(), self.cell_state.cuda()))
                
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
        
        if self.scalar is True:
            if self.bidirectional is True:
                #   stack the hidden_states of the last reverse layer [-1] (leftmost) and the
                #   last forward layer[-2] (rightmost)
                final_hidden_state = torch.cat((final_hidden_states[-1], 
                                                final_hidden_states[-2]), dim=1)
            else:
                final_hidden_state = final_hidden_states[-1]
            y_pred = F.relu(self.linear(final_hidden_state))
            #y_pred = torch.mul(y_pred, torch.tensor(seq_lens).unsqueeze(1).float().cuda())          #   scale to number of errors from error rate

        else:
            #   bidirectional lstm is already concatenated
            y_pred = F.softmax(self.linear(lstm_out), dim=2)
        
        return y_pred
        
    def compute_loss(self, y, target, seq_lens=[]):
        """
        Compute the loss function using the MSE Loss function
        Arguments:
            y (torch.tensor) : (padded) outputs of classifier for input (batch_size, output_dim) 
                (self.scalar=True) or (max(seq_len), batch_size, output_dim) (self.scalar=False)
            target (torch.tensor) : (padded) ground truth labels for input (batch_size, output_dim)
                (self.scalar=True) or (max(seq_len), batch_size, output_dim) (self.scalar=False)
            seq_lens (list[int]) : seq_len of each sample in output (only for self.scalar=False)
        Returns:
            loss (torch.tensor) : MSE loss of y (self.scalar=True)
        """
        if self.scalar is True:          
            return F.mse_loss(y, target)
        else:
            if len(seq_lens) != y.shape[1] or y.shape[1] != target.shape[1]: 
                
                print("Error: Batch size mismatched!!!")
                print("len(seq_lens): ", len(seq_lens))
                print("y.shape[1]: ", y.shape[1])
                print("target.shape[1]: ", target.shape[1])
                
                return torch.tensor([-1])       #   causes an error :D
            
            for idx in range(0, y.shape[1]):         #   for every sample in batch
                #   remove padding and flatten to 1D                
                y_tensor1D = y[:seq_lens[idx], idx, :].flatten()
                target_tensor1D = target[:seq_lens[idx], idx, :].flatten()
                
                #   concatenate into 1D
                if idx == 0:        #    if first sample
                    y_flatten = y_tensor1D
                    target_flatten = target_tensor1D
                else:
                    y_flatten = torch.cat((y_flatten, y_tensor1D))
                    target_flatten = torch.cat((target_flatten, target_tensor1D))
            
            return F.binary_cross_entropy(y_flatten, target_flatten)


    def tensors2packedseq(self, tensors, seq_lens):
        """
        Converts an ordered 3d padded tensor (with ORIGINALLY variable sequence lengths) to a PackedSequence
        datatype)
        Arguments:
            tensors (torch.tensor) : padded input tensor of shape (batch_size, max_seq_len, input_dim)
            seq_lens (list[int]) : unpadded sequence length of each tensor in tensors[a, :, :]
        Returns:
            packedseq (PackedSequence) : a PackedSequence type of batch of input tensors 
        """

        #   convert 3d tensor to list of 2d tensors
        tensor_list = []
        for idx in range(tensors.shape[0]):                    #   for every sample in batch
            tensor_list.append(tensors[idx, :, :])

        packedseq = nn.utils.rnn.pad_sequence(tensor_list)  #   tensor (seq_len, batch_size, input_dim)
        packedseq = nn.utils.rnn.pack_padded_sequence(packedseq, seq_lens)
    
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

def lists2onehottensors(lists, dim, onehot_encoder, pad_token=0):
    """
    Converts padded list of lists of integers to 3d tensor (max(seq_len), batch_size, dim)
    Arguments:
        lists (list[list[int]]) : list of integers lists with all internal lists of equal length
        onehot_encoder (class OneHotEncoder) : encoder for 1-hot encoding using OneHotEncoder
        dim (int) : number of output dimensions for 1 hot tensor
        pad_token (int) : integer value which indicates an empty 1-hot vector (default 0)
    Returns:
        tensor (torch.tensor) : padded 3d tensor of shape (max(seq_len),, batch_size, dim)
    """        
    tensor_list = []
    
    for sample_idx in range(0, len(lists)):       #   for every sample in the batch
        sample = lists[sample_idx]
        sample = np.array(sample)       #   convert to np.array
        sample = sample.reshape(len(sample), 1)
        encoded_sample = torch.tensor(onehot_encoder.fit_transform(sample)).float().cuda()
        tensor_list.append(encoded_sample)   #  shape(seq_len, dim)
    
    tensors = torch.stack(tensor_list)          #   shape(batch_size, seq_len, dim)
    #   reshape to (seq_len, batch_size, dim)
    tensors = tensors.transpose(dim0=0, dim1=1)
        
    return tensors    

def onehottensors2classlist(onehottensor, seq_lens):
    """
    Converts a 3d tensor (seq_len, batch_size, output_dim) to a 2d class list (list[batch_size  * list[seq_len]])
        where each class is a unique integer
    Arguments:
        onehottensor (torch.tensor) : 3d padded one-hot tensor of different sequence lengths of
            shape (max(seq_lens), batch_size, output_dim)
        seq_lens (list[int]) : length of each of the sequences without padding 
            (i.e. onehottensor.shape[1] without padding)
    Returns:
        batch_list (list[list[int]]) : list of class lists with each internal list corresponding
            to a sequence and each class in the internal list corresponding to a class for 
            each step of the sequence
    """
    batch_list = []
    
    for idx in range(0, onehottensor.shape[1]):     # for every tensor sample in batch
        integer_list = []
        tensor2d = onehottensor[:, idx, :]          # shape (seq_len, dim)
        max_classes = tensor2d.max(dim=1)
        integer_list = max_classes[1].tolist()
        
        batch_list.append(integer_list)
    
    return batch_list

def pad_lists(lists, pad_token=0, seq_lens_idx=[]):
    """
    Pads unordered lists of different lengths to all have the same length (max length) and orders
    length descendingly
    Arguments:
        lists : list of 1d lists with different lengths (list[list[int]])
        pad_token : padding value (int)
        seq_lens_idx : list of sorted indices (list[int])
    Returns:
        ordered_lists (list[list[int]]) : List of padded 1d lists with equal lengths, ordered descendingly
                from original lengths
        ordered_seq_lens (list[int]) : List of sequence lengths of corresponding lists (i.e. len(lst))
        seq_lens_idx (list[int]) : Order of original indices in descending sequence length
    """
    
    seq_lens = [len(lst) for lst in lists]
        
    max_seq_len = max(seq_lens)
    ordered_seq_lens = []
    ordered_lists = []

    if len(seq_lens_idx) == 0:
        seq_lens_idx = np.flip(np.argsort(seq_lens), axis=0).tolist()     # descending indices based on seq_lens    
    
    for idx in seq_lens_idx:                    #   for every sample in batch
        ordered_lists.append(lists[idx] + [pad_token] * (max_seq_len - len(lists[idx])))
        ordered_seq_lens.append(seq_lens[idx])
    
    return ordered_lists, ordered_seq_lens, seq_lens_idx


def train(clf, onehot_encoder, params):
    """
    Trains the model given training data.
    Arguments:
        clf (class) : lstm model
        params (dict) : contains model, dimension, and batch parameters
    Returns:
        clf (class) : the trained model
    """
    num_epochs = params['num_epochs']
    lr = params['learning_rate']
    scalar = params['scalar']
    output_dim = params['output_dim']
    filename = params['save_name']
    bestname = params['best_name']
    
    #   name report file for writing
    train_filename = SAVE_DIR + str(dt.datetime.now()) + "_" + params['train_report']
    
    #   Create training data DataLoader
    dl_full = DataLoader(DATA_DIR, PHONEME_OUT, params['align'], 'full', 'val',
                            batch_size = params['batch_size'])
    full_dict = dl_full.int_to_phoneme
    
    dl_train = DataLoader(DATA_DIR, PHONEME_OUT, params['align'],
                    params['data_type'], 'train', batch_size = params['batch_size'])
    dl_val = DataLoader(DATA_DIR, PHONEME_OUT, params['align'],
                    params['data_type'], 'val', batch_size = params['batch_size'])
    int_to_pho_dict = dl_val.int_to_phoneme
    
    #   Create optimizer

    optimizer = torch.optim.Adam(list(clf.parameters()), lr=lr)

    #   consistent validation case

    X_fixed_batch, y_fixed_batch = dl_val.__iter__().__next__()                 #   gets a single batch
    X_fixed = X_fixed_batch[2]
    y_fixed = y_fixed_batch[2]
    
    y_pred = predict(X_fixed, y_fixed, clf, onehot_encoder, params)

    X_fixed_phoneme = int2phoneme(full_dict, X_fixed)
    y_fixed_phoneme = int2phoneme(int_to_pho_dict, y_fixed)
    y_pred = int2phoneme(int_to_pho_dict, y_pred)
    print("X_fixed: ", X_fixed_phoneme)
    print("y_fixed: ", y_fixed_phoneme)
    print("y_pred: ", y_pred)

    #   best model parameters
    best_loss = 9999999999999           #   infinitely high
    best_epoch = -1

    for epoch in range(0, num_epochs):
        try:
            batch_idx = 0        
            print("Epoch: " + str(epoch) + "     " + str(dt.datetime.now()))
            
            sum_train_loss = 0
            sum_train_iters = 0
            
            #   open report file for writing
            train_file = open(train_filename, "a+")
            
            for X_train, y_train in dl_train:
                #print("Batch Idx: ", batch_idx, "     ", dt.datetime.now())
                X_train_batch, X_train_seq_lens, seq_lens_idx = pad_lists(X_train)
                X_train_batch = torch.tensor(X_train_batch).cuda()   
                        
                if scalar is True:
                    #   sort y_train_batch
                    y_train_batch = torch.tensor([[y_train[idx]] for idx in seq_lens_idx]).float().cuda()
                else:
                    #   convert list[list[int]] to padded 3d tensor (seq_len, batch_size, output_dim)                
                    y_train_batch, y_train_seq_lens, seq_lens_idx = pad_lists(y_train, 
                                                                              seq_lens_idx=seq_lens_idx)
                    y_train_batch = lists2onehottensors(y_train_batch, output_dim, onehot_encoder,
                                                        pad_token=0)
                
                #   Zero the gradient of the optimizer
                optimizer.zero_grad()
                
                #   Forward pass
                y_pred = clf.forward(X_train_batch, X_train_seq_lens)
                loss = clf.compute_loss(y_pred, y_train_batch, X_train_seq_lens)
            
                sum_train_loss += loss.item()
                sum_train_iters += 1
            
                loss.backward()     #   backward pass
                optimizer.step()    #   update gradient step
                
                batch_idx += 1
    
            #   Average training loss
            train_loss = float(sum_train_loss) / sum_train_iters
    
            #   Validation Loss
            val_loss, y_pred, X_val, y_val = evaluate(dl_val, clf, onehot_encoder, params)
            
            print("Training Loss: " + str(train_loss) + "    Val Loss: " + str(val_loss))
            train_file.write("Epoch: " + str(epoch) + "    Training Loss: " + str(train_loss) + 
                             "    Val Loss: " + str(val_loss) + "\n")
            train_file.close()          #   close file after writing to save
            
            #   print fixed validation case
            y_pred = predict(X_fixed, y_fixed, clf, onehot_encoder, params)
            y_pred = int2phoneme(int_to_pho_dict, y_pred)
            print(y_pred)
    
            #   save progress
            if epoch % 2 == 0:
                checkpoint = {'model': lstm_rnn(params),
                              'state_dict': clf.state_dict(),
                              'optimizer': optimizer.state_dict()}
                file =  SAVE_DIR + filename + "_" + str(epoch) + ".pth"
                torch.save(checkpoint, file)
                
            if best_loss > val_loss:
                best_checkpoint = {'model': lstm_rnn(params),
                              'state_dict': clf.state_dict(),
                              'optimizer': optimizer.state_dict()}
                best_epoch = epoch
                best_loss = val_loss
        
        except KeyboardInterrupt:
            print("Exiting training loop early...")
            #   save best checkpoint
            file =  SAVE_DIR + bestname + "_" + str(best_epoch) + ".pth"
            torch.save(best_checkpoint, file)
            print("Best Val Loss at Epoch " + str(best_epoch) + ": " + str(best_loss))
            break
    
    #   save best checkpoint
    file =  SAVE_DIR + bestname + "_" + str(best_epoch) + ".pth"
    torch.save(best_checkpoint, file)
    print("Best Val Loss at Epoch " + str(best_epoch) + ": " + str(best_loss))
        
    return clf

def evaluate(dataloader, clf, onehot_encoder, params):
    
    scalar = params['scalar']
    output_dim = params['output_dim']
    
    sum_loss = 0
    num_iters = 0
    
    for X_test, y_test in dataloader:
    
        X_test_batch, X_test_seq_lens, seq_lens_idx = pad_lists(X_test)
        X_test_batch = torch.tensor(X_test_batch)
        
        if scalar is True:
            #   sort y_train_batch
            y_test_batch = torch.tensor([[y_test[idx]] for idx in seq_lens_idx]).float().cuda()
        else:
            #   convert list[list[int]] to padded 3d tensor (seq_len, batch_size, output_dim)
            y_test_batch, y_test_seq_lens, seq_lens_idx = pad_lists(y_test, 
                                                                    seq_lens_idx=seq_lens_idx)        
            y_test_batch = lists2onehottensors(y_test_batch, output_dim, onehot_encoder,
                                               pad_token=0)
            
        y_pred = clf.forward(X_test_batch, X_test_seq_lens)
        loss = clf.compute_loss(y_pred, y_test_batch, X_test_seq_lens)
        sum_loss += loss.item()
        num_iters += 1
        
        if scalar is False:
            #   convert softmax y_pred and y to 1d list
            y_test_batch = onehottensors2classlist(y_test_batch, X_test_seq_lens)
            y_pred = onehottensors2classlist(y_pred, X_test_seq_lens)
    
    loss = sum_loss / num_iters
    
    return loss, y_pred, X_test_batch, y_test_batch

def predict(X, y, clf, onehot_encoder, params):
    """
    Runs a forward pass for a SINGLE sample and returns the output prediction.
    Arguments:
        X (list[int]) : a list of integers with each integer an input class of step
        y (list[int]) : a list of integers with each integer an output class of step
    Returns:
        y_pred (list[int]) : a list of integers with each integer the prediction of each step
    """
    scalar = params['scalar']
    output_dim = params['output_dim']
    
    X = torch.tensor(X).cuda()         #   shape(seq_len,)
    X = X.unsqueeze(0)          #   shape(batch_size=1, seq_len)
    
    if scalar is True:
        seq_len = [X.shape[1]]
    else:
        seq_len = [len(y)]              #   2d list
    
    if scalar is True:
        y = torch.tensor([[y]]).cuda().float().cuda()         #   shape(1,)
    else:
        y = lists2onehottensors([y], output_dim, onehot_encoder, pad_token=0)
        #   change to 1-hot

    y_pred = clf.forward(X, seq_len)
    loss = clf.compute_loss(y_pred, y, seq_len)
    loss = loss.item()
    
    if scalar is False:
            #   convert softmax y_pred and y to 1d list
            y_pred = onehottensors2classlist(y_pred, seq_len)[0]
    
    return y_pred

    

def create_sample_data(params, min_seq_len=3, max_seq_len=10, scalar=True, shuffle=False):
    """
    **Don't use this if you don't know it.  Buggy and only for verifying code runs.**
    
    Creates data X, y for a very simple test case.  For first half of the batch
    will be labelled 1 and will have X < vocabSize/2.  The second half of the batch will
    be labelled 0 and will have X typically > vocabSize/2.
    Arguments:
        params (dict) : contains dimension and batch parameters
        min_seq_len (int) : length of the shortest possible sequence
        max_seq_len (int) : length of the longest possible sequence
        scalar (boolean) : whether y is a binary output at the end of the entire sequence (True)
            or y is seq_len one-hot vectors (i.e. every sequence step has an output) (False)
        shuffle (boolean) : whether to keep samples ordered by descending seq_len (False) or not (True)
    Returns:
        List of integer sequences in X (List[list[int]])
        torch.tensor(batch_size, output_dim) : y labels corresponding to X
    """

    vocabSize = params['embed_dim']
    output_dim = params['output_dim']
    batch_size = params['batch_size']
           
    #   generate random sorted seq_len
    seq_len = torch.randint(min_seq_len, max_seq_len, (batch_size,))      
    
    if shuffle is False:
        print("Shuffle is false.")
        seq_len = seq_len.sort(descending=True)[0]
    
    #   create y
    if scalar is True:
        y = torch.ones((batch_size,output_dim))
        y[int(batch_size/2):, ] = 0
    else:
        #   create list of lists of integers y of different sequence lengths
        y = []
        for i in range(0, batch_size):
            if i > int(batch_size/2)-1:
                sequence = list(range(0, seq_len[i]))      #   ascending consecutive integers
            else:
                sequence = list(range(seq_len[i], 0, -1))  #   descending consecutive integers
            #   make sure all values in sequence is less than output_dim
            sequence = [value if value < output_dim else output_dim-1 for value in sequence]

            y.append(sequence)
    
    #   create data sequences X of different sequence lengths
    X = []
    for i in range(0, batch_size):
        if i > int(batch_size/2)-1:
            sequence = [random.randrange(1, int(vocabSize/2)) for _ in range(seq_len[i])]
        else:
            sequence = [random.randrange(int(vocabSize/2), vocabSize) for _ in range(seq_len[i])]
        X.append(sequence)

    return X, y

def load_checkpoint(filename):
    """
    Loads a saved model
    Arugments:
        filename (str) : name of file in SAVE_DIR
    Returns
        model : saved model in filename
    """
    checkpoint = torch.load(SAVE_DIR + filename)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    #for parameter in model.parameters():
    #    parameter.requires_grad = False
    #model.eval()
    return model
    

def main(params, load_model=False, train_model=True, verbose=False):
    
    scalar = params['scalar']
    output_dim = params['output_dim']
    
    #   initialize classifier
    if verbose:
        print("Initializing classifier...")
    
    #   checks
    if scalar is True and params['output_dim'] != 1:
        print("WARNING: Scalar is True but output dim is NOT set to 1...")
        print("Automatically setting to 1...")
        params['output_dim'] = 1
    
    #   Create model
    if load_model is True:
        print("Loading model...")
        clf = load_checkpoint(params['load_name']).cuda()
    else:
        print("Initializing model...")
        clf = lstm_rnn(params).cuda()
    
    #   Create one-hot encoder
    onehot_encoder = OneHotEncoder(output_dim, sparse=False)
    
    #   Train model    
    if train_model is True:
        clf = train(clf, onehot_encoder, params)
    
    #   Evaluate model - outputs X_test,  y_test because ordered inside of evaluate
    
    dl_test = DataLoader(DATA_DIR, PHONEME_OUT, params['align'],
                    params['data_type'], 'test', batch_size = params['batch_size'])
    
    #   Test Loss
    loss, y_pred, X_test, y_test = evaluate(dl_test, clf, onehot_encoder, params)
    
    #   Test Report
    test_file = open(SAVE_DIR + str(dt.datetime.now()) + "_" 
                          + params['test_report'], "w")
    
    if scalar is True:
        for sample in range(0, len(y_test)):
            print("Label: ", float(y_test[sample]), "    Predict: ", float(y_pred[sample]))
            test_file.write("Label: " + str(float(y_test[sample])) + "    Predict: " +
                            str(float(y_pred[sample])))
        print("Loss: " + str(loss))
        
    else:
        for sample in range(0, len(y_test)):
            print("Label: ", y_test[sample])
            print("Predict: ", y_pred[sample])
            test_file.write("Label: " + str(y_test[sample]))
            test_file.write("Predict: " +  str(y_pred[sample]))
        print("Loss: ", str(loss))
    
    test_file.write("Loss: " + str(loss))
    test_file.close()
    
if __name__ == '__main__':
    
    #torch.manual_seed = 1
    alignment = 'hypothesis'
    data_type = 'scalar'

    
    #   get input dim (always 44)
    dl_1 = DataLoader(DATA_DIR, PHONEME_OUT, alignment, 'full', 'val', batch_size=4)
    input_dim = dl_1.vocab_size
    
    #   get output dim (e.g. binary = 3, full = 44)
    dl_2 = DataLoader(DATA_DIR, PHONEME_OUT, alignment,
                    data_type, 'val', batch_size = 4)
    output_dim = dl_2.vocab_size
    
    params = {
            'input_dim':        input_dim,
            'embed_dim':        input_dim,
            'hidden_dim':       25,            
            'output_dim':       output_dim,              
            'num_layers':       1,
            'batch_size':       256,
            'num_epochs':       200,
            'learning_rate':    0.05,
            'learning_decay':   0.9,
            'bidirectional':    False,
            'scalar':           True,
            'align':            alignment,
            'data_type':        data_type,
            'load_name':        'lstm_10.pth',
            'save_name':        'lstm',
            'best_name':        'lstm_best',
            'train_report':     'lstm_report.txt',
            'test_report':      'lstm_test.txt',
    }

    main(params, load_model=False, train_model=True, verbose=True)
