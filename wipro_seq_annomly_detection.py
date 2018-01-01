
# coding: utf-8

# # WIPRO TIME SERIERS - ANOMALY DETECTION AND SEQUENCE PREDICTION

# **Requirements**

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

#get_ipython().magic(u'matplotlib inline')
#get_ipython().magic(u'env CUDA_VISIBLE_DEVICES=1')

USE_CUDA = False #torch.cuda.is_available()


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
        
        print(hidden.size())
        # convert to batch first
        M = M.permute(1,0,2)
        hidden = hidden.permute(1,0,2)

        energies = self._score(hidden, M)
        a = F.softmax(energies)
        print('a size {} M size {}'.format(a.size(), M.size()))
        c = a.permute(0,2,1).bmm(M) #bmm(a.permute(0,2,1))
        print('c size {} hidden size {}'.format(c.size(), hidden.size()))
        next_timestep_prediction = self.fc_out(torch.cat((c.squeeze(1), hidden.squeeze(1)), 1))
        
        return next_timestep_prediction, a
        
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



# In[4]:


class SeqRnnAttnAndPred(nn.Module):
    """
    SequenceAttnPred - Recurrent Atteniton based model for predicting next time step.
    """
    def __init__(self, input_size, hidden_size, output_size, batch_size,
                 rnn_layers=1, atnn_method='general', memory_size=-1):
        """
        args:
        input_size: size of elemnt of sequnece.
        hidden_size: size of hidden state of RNN (same as output if only 1 RNN).
        output_size: size of output tensor.
        rnn_layers: amount of stacked RNN's (see any basic seq2seq paper).
        atnn_method: type of attention to use.
        memory_size: amount of rnn output's to aggregate and attend over.
        """
        super(SeqRnnAttnAndPred, self).__init__()
        self._rnn_layer = SequencePredRNN(input_size, hidden_size, rnn_layers)
        self._atnn_layer = AttendAndPredict(atnn_method, hidden_size, output_size)
        self._memory_size = memory_size
        self._batch_size = batch_size
        self._hidden = Variable(torch.zeros(1, batch_size, hidden_size))
        self._hidden = self._hidden.cuda() if USE_CUDA else self._hidden
        
    def forward(self, inputs):
        """
        args:
        inputs(seq_len,batch,input_len): input sequence predict seq_len + 1
        """
        seq_len = inputs.size()[0]
        
        # in case of 1 rnn layer output == hidden.
        seqrnn_output, self._hidden = self._rnn_layer(inputs, self._hidden)
        self.memory = seqrnn_output[-self._memory_size-1:-1]
        self._hidden = seqrnn_output[-1].unsqueeze(0)
        outputs, alignment = self._atnn_layer(self._hidden, self.memory)
        return outputs, alignment
   


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


# ** Training procedure**

# Train per time window

# In[6]:


def train(input_seq, model, criterion, optimizer):
    """
    Train for a given sequence batch size.
    Args:
    input_seq(batch_size,seq_len,input_size): tensor containin sequences.
    model: input model - batch is first dim.
    criterion: distance measure i.e. l1, l2 etc.
    optimizer: GD, ADAM etc.
    """  
    preds, alignment = model(input_seq[:-1])
    loss = criterion(preds, input_seq[-1])
    print(loss)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.data[0]
        


# This is a helper function to print time elapsed and estimated time remaining given the current time and progress %.

# Full train procedure

# In[7]:


def trainIters(train_data, model, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().cuda()

    for iter in range(1, n_iters + 1):
        
        input_variable = train_data[iter,:].unsqueeze(-1).unsqueeze(-1)
        double_batch_input = torch.cat((input_variable,input_variable),1)
        
        loss = train(double_batch_input, model, criterion, optimizer)
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


# In[8]:


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

# In[9]:


import numpy as np

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'float32')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = Variable(torch.from_numpy(np.sin(x / 1.0 / T).astype('float32')), requires_grad=False)
data = data.cuda() if USE_CUDA else data


# In[10]:


print('example data shape: '.format(data.size()))
show_plot(data[1,:].data.cpu().numpy())


# ** Main Entry Point **

# In[11]:


hidden_size = 1
seq_attn_pred = SeqRnnAttnAndPred(input_size=1, hidden_size=1, output_size=1, batch_size=2,
                  rnn_layers=1, atnn_method='general', memory_size=10)

if USE_CUDA:
    seq_attn_pred = seq_attn_pred.cuda()

trainIters(data, seq_attn_pred, 75000, print_every=5000)


# In[ ]:




