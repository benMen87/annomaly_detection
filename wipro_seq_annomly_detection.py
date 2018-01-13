
# coding: utf-8

# # WIPRO TIME SERIERS - ANOMALY DETECTION AND SEQUENCE PREDICTION

# **Requirements**

# In[1]:

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

#get_ipython().magic(u'matplotlib inline')
#get_ipython().magic(u'env CUDA_VISIBLE_DEVICES=0')

USE_CUDA = torch.cuda.is_available()
os.environ['env CUDA_VISIBLE_DEVICES']='1'


# **Build recurrent anttention based model**
# 
# The idea being learn the sequential dependencys and structure via gated rnn,
# and to learn spatial structure and similarity via attention mechanism. 
# 
# The Attention used in this case is inspired by https://arxiv.org/pdf/1508.04025.pdf
# Where the context vector is appended to final hiddent state and are passed to output layer.
# This differs to the standerd case where context is appended to input of the reccurent cell.
# 
# The RNN sequence is inspired by https://arxiv.org/pdf/1308.0850.pdf
# Where LSTM based models are used for pedicting next time step.
# 
# The code is mostly based on code found here:
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# ** Recurrent model for capturing sequential information**

# In[2]:


class SequencePredRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(SequencePredRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False)

    def forward(self, inputs, hidden):
        output = inputs
        for i in range(self.n_layers):
            print('outshape {} hiddenshape {}'.format(inputs.size(), hidden.size()))
            output, hidden = self.gru(output, hidden)
        return output, hidden


# ** Attention model utilize reccurnt ouputs and their spatial similarity for predicting next time step**

# In[3]:


class AttendAndPredict(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, method, hidden_size, output_size):
        super(AttendAndPredict, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define layers
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))
            
        self.fc_out = nn.Linear(self.hidden_size * 2, self.output_size)
        
        

    def forward(self, hidden, M):
        """Attend over N rnn sequence prediction outputs till time t-1 (t-2-N...t-1).
        
        After creating variables to store the attention energies, calculate their 
        values for each encoder output and return the normalized values.
        
        Args:
            M(seq_len,batch,input size): memory of which to attend over.
            hidden(1, batch, input size): hidden state.
            
        Returns:
             Normalized (0..1) energy values, re-sized to 1 x 1 x seq_len
        """
        
        seq_len = M.size()[0]
        batch_size = M.size()[1]

        # convert to batch first
        M = M.permute(1,0,2)
        hidden = hidden.permute(1,0,2)

        if USE_CUDA:
            energies = Variable(torch.zeros(batch_size, seq_len)).cuda()
        else:
            energies = Variable(torch.zeros(batch_size, seq_len))
        
        energies[:] = self._score(hidden, M)
            
        #for i in range(seq_len):
        #    energies[i] = self._score(hidden, M[i])
        a = F.softmax(energies).unsqueeze(1)
        c = a.bmm(M)
        next_timestep_prediction = self.fc_out(torch.cat((c.squeeze(1), hidden.squeeze(1)), 1))
        
        return next_timestep_prediction
        
    def _score(self, hidden, M):
        """
        Calculate the relevance of a particular encoder output in respect to the decoder hidden.
        Args:
            hidden: decoder hidden output used for condition.
            M(batch,seq_len,input_size): memory of which to attend over.
            hidden(1, batch, input size): hidden state.
        """

        if self.method == 'dot':
            # TODO: Not tested
            energy = hidden.dot(M)
        elif self.method == 'general':
            energy = self.attention(M)
            energy = torch.bmm(energy, hidden)#hidden.dot(energy)
        elif self.method == 'concat':
            # TODO: Not tested
            energy = self.attention(torch.cat((hidden, M), 1))
            energy = self.other.dor(energy)
        return energy


# ** Training procedure**

# Train per time window

# In[4]:


def train(input_seq, seqrnn_model, atten_model, memory_size, criterion, optimizer):
    """
    Train for a given sequence batch size.
    Args:
    input_seq(batch_size,seq_len,input_size): tensor containin sequences.
    seqrnn_model: input model - batch is first dim.
    memory_size: size of matrix to attend over.
    criterion: distance measure i.e. l1, l2 etc.
    optimizer: GD, ADAM etc.
    """

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result
   
    seq_len = input_seq.size()[0] 
    batch_size = input_seq.size()[1]
    input_len = input_seq.size()[2]

    assert memory_size < seq_len, 'memory should be smaller than total seq'

    seq_rnn_hidden = Variable(torch.zeros(1, batch_size, input_len))
    seqrnn_output = Variable(torch.zeros(seq_len, batch_size, seqrnn_model.hidden_size))
    seq_rnn_hidden = seq_rnn_hidden.cuda() if USE_CUDA else seq_rnn_hidden
    seqrnn_model = seqrnn_model.cuda() if USE_CUDA else seqrnn_model
    
    
    seqrnn_model.zero_grad()
    atten_model.zero_grad()
    loss = 0
    
    #
    # Start prediction from time t + memory_size till time T - memory_size
    seqrnn_output[:], seq_rnn_hidden = seqrnn_model(input_seq, seq_rnn_hidden) # in case of 1 rnn layer output == hidden.
    for t in range(seq_len - memory_size):
        M = seqrnn_output[t:memory_size+t]# memory matrix i.e. attend over
        h = seqrnn_output[memory_size+t].unsqueeze(0) # current rnn pred
        preds = atten_model(h, M)
        loss += criterion(preds, input_seq[memory_size+t])
        
    loss.backward()
    optimizer.step()
    
    return loss.data[0] / ((seq_len - memory_size)*batch_size)
        


# This is a helper function to print time elapsed and estimated time remaining given the current time and progress %.

# In[5]:


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# Full train procedure

# In[6]:


def trainIters(train_data, recurrent_model, attention_model, n_iters,
               print_every=1000, plot_every=100, learning_rate=0.01):
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.SGD(nn.ModuleList([recurrent_model, attention_model]).parameters(), lr=learning_rate)
    training_pairs = [] # TODO: Need to add functionatly here
    criterion = nn.L1Loss()

    for iter in range(1, n_iters + 1):
        
        input_variable = train_data[iter,:].unsqueeze(-1).unsqueeze(-1)

        loss = train(torch.cat((input_variable,input_variable),1), recurrent_model,
                     attention_model, 10, criterion, optimizer)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


# In[7]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# Temprorery gen syntetic data for debugging model

# In[8]:


import numpy as np

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = Variable(torch.from_numpy(np.sin(x / 1.0 / T).astype('float32')), requires_grad=False)
data = data.cuda() if USE_CUDA else data

# In[9]:


print('example data')
show_plot(data[1,:].data.cpu().numpy())


# ** Main Entry Point **

# In[10]:


hidden_size = 1
seqrnn = SequencePredRNN(1, 1)
attn = AttendAndPredict('general', 1, 1)

if USE_CUDA:
    seqrnn = seqrnn.cuda()
    attn = attn.cuda()

trainIters(data, seqrnn, attn, 75000, print_every=5000)
