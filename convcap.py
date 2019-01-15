import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def Conv1d(in_channels, out_channels, kernel_size, padding, dropout = 0):
    params = nn.Conv1d(in_channels, out_channels, kernel_size, padding = padding)
    std = math.sqrt(4 * (1 - dropout) / (kernel_size * in_channels))
    params.weight.data.normal_(mean = 0, std = std)
    params.bias.zero_()
    return params

def Embedding(num_embeddings, embedding_dim, padding_idx):
    params = nn.Embedding(num_embeddings, embedding_dim, padding_idx = padding_idx)
    params.weight.data.normal_(0, 0.1)
    return params

def Linear(in_features, out_features, dropout = 0.):
    params = nn.Linear(in_features, out_features)
    std = math.sqrt((1 - dropout) / (in_features))
    params.weight.data.normal_(mean = 0, std = std)
    params.bias.data.zero_()
    return params

class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embedding_dim):
        super(AttentionLayer, self).__init__()
        self.in_projection = Linear(conv_channels, embedding_dim)
        self.out_projection = Linear(embedding_dim, conv_channels)
        self.bmm = torch.bmm
    
    def forward(self, x, wordemb, imgsfeatures):
        '''
        x is the input of CNN without any convolution
        '''
        residual = x
        x = (self.in_projection(x) + wordemb) * math.sqrt(0.5)
        b, c, f_h, f_w = imgsfeatures.size()
        y = imgsfeatures.view(b, c, f_h, f_w)

        x = self.bmm(x, y)

        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]))
        x = x.view(sz)
        attn_scores = x

        y = y.permute(0, 2, 1)

        x = self.bmm(x, y)

        s = y.size(1)
        x = x * (s * math.sqrt(1.0 / s))

        x = (self.out_projection(x) + residual) * math.sqrt(0.5)

        return x, attn_scores



class convcap(nn.Module):
    def __init__(self, num_wordclass, kernel_size = 5, num_layers = 1, is_attention = True, nfeatures = 512, dropout = 0.1):
        super(convcap, self).__init__()
        self.nimgfeatures = 4096
        self.is_attention = is_attention
        self.nfeatures = 512
        self.dropout = dropout

        self.emb_0 = Embedding(num_wordclass, nfeatures, padding_idx = 0)
        self.emb_1 = Linear(nfeatures, nfeatures, dropout = dropout)

        self.imgproj = Linear(self.nimgfeatures, self.nfeatures, dropout= dropout)
        self.resproj = Linear(nfeatures * 2, self.nfeatures, dropout = dropout)

        n_in = 2 * self.nfeatures
        n_out = self.features

        self.n_layers = num_layers
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.kernel_size = kernel_size
        self.pad = self.kernel_size - 1
        for i in range(self.n_layers):
            self.convs.append(Conv1d(n_in, 2 * n_out, self.kernel_size, self.pad, dropout))
            if (self.is_attention):
                self.attention.append(AttentionLayer(n_out, nfeatures))
            n_in = n_out
        self.classifier_0 = Linear(self.nfeatures, (nfeatures // 2))
        self.classifier_1 = Linear((nfeatures // 2), num_wordclass, dropout = dropout)

    def forward(self, imgsfeatures, imgsfc7, wordclass):
        atten_buffer = None
        wordemb = self.emb_0(wordclass)
        wordemb = self.emb_1(wordemb)
        x = wordemb.transpose(2, 1)
        batchsize, wordembdim, maxtokens = x.size()

        '''
        image feature after fc7 will be embedded as vector, and then cancatenate with 
        word vector to form the input of caption CNN
        '''
        y = F.relu(self.imgproj(imgsfc7))
        y = y.unsqueeze(2).expand(batchsize, self.nfeatures, maxtokens)
        
        x = torch.cat([x, y], 1) # tensor x will be the input of caption CNN

        for i, conv in enumerate(self.convs):
            if i == 0:
                x = x.transpose(2, 1)
                residual = self.resproj(x)
                residual = residual.transpose(2, 1)
                x = x.tranpose(x)
            else :
                residual = x

            x = F.dropout(x, p = self.dropout, training = self.training)
            x = conv(x)
            x = x[:, :, :-self.pad]

            x = F.glu(x, dim = 1)

            if self.is_attention:
                attn = self.attention[i]
                x = x.tranpose(2, 1)
                x, attn_buffer = attn(x, wordemb, imgsfeatures)
                x = x.tranpose(2, 1)
            x = (x + residual) * math.sqrt(0.5)

        x = x.transpose(2, 1)
        x = self.classifier_0(x)
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = self.classifier_1(x)
        x = x.transpose(x)
        return x, attn_buffer

    
