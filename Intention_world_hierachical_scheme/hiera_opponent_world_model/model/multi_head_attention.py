import torch
from torch import nn
import math
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_dim, num_heads, n_time_sequence):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.n_time_sequence=n_time_sequence
        self.register_buffer("mask",torch.tril(torch.ones((n_time_sequence+1,n_time_sequence+1)))
                             .view(1,1,n_time_sequence+1,n_time_sequence+1))

    def forward(self,query,key,value):
        # N,T,D
        N,T,D=query.size()
        q=query.view(N, T, self.num_heads, D//self.num_heads).transpose(1,2)
        k = key.view(N, T, self.num_heads, D // self.num_heads).transpose(1, 2)
        v = value.view(N, T, self.num_heads, D // self.num_heads).transpose(1, 2)

        attention=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        attention=attention.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att=F.softmax(attention,dim=-1)

        y=att@v
        y=y.transpose(1,2).reshape(N,T,D)
        return y

class MultiHeadAttention_GlobalSpace(nn.Module):
    def __init__(self,hidden_dim, num_heads, n_time_sequence):
        super(MultiHeadAttention_GlobalSpace, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=4*num_heads
        self.n_time_sequence=n_time_sequence
        # self.register_buffer("mask",torch.tril(torch.ones((n_time_sequence+1,n_time_sequence+1)))
        #                      .view(1,1,n_time_sequence+1,n_time_sequence+1))

    def forward(self,query,key,value):
        # N,T*D
        N,D=query.size()
        q=query.view(N, self.num_heads, D//self.num_heads).transpose(1,0)
        k = key.view(key.shape[0],  self.num_heads, key.shape[-1]// self.num_heads).transpose(1, 0)
        v = value.view(value.shape[0], self.num_heads, value.shape[-1] // self.num_heads).transpose(1, 0)

        attention=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        # attention=attention.masked_fill(self.mask[:,:,,:T,:T]==0, float('-inf'))
        att=F.softmax(attention,dim=-1)

        y=att@v
        y=y.transpose(1,0).reshape(N,-1)
        return y

    def forward2(self,query,key,value):
        # B, N,T*D
        B,N,D=query.size()
        q=query.view(B,N, self.num_heads, D//self.num_heads).transpose(1,0)
        k = key.view(key.shape[0],key.shape[1],  self.num_heads, key.shape[-1]// self.num_heads).transpose(2, 1)
        v = value.view(value.shape[0], value.shape[1],self.num_heads, value.shape[-1] // self.num_heads).transpose(2, 1)

        attention=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        # attention=attention.masked_fill(self.mask[:,:,,:T,:T]==0, float('-inf'))
        att=F.softmax(attention,dim=-1)

        y=att@v
        y=y.transpose(2,1).reshape(B,N,-1)
        return y

class MultiHeadAttention_GlobalTime(nn.Module):
    def __init__(self,hidden_dim, num_heads, n_time_sequence):
        super(MultiHeadAttention_GlobalTime, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=4*num_heads
        self.n_time_sequence=n_time_sequence
        self.register_buffer("mask",torch.tril(torch.ones((n_time_sequence+1,n_time_sequence+1)))
                             .view(1,n_time_sequence+1,n_time_sequence+1))
        self.register_buffer("mask2", torch.tril(torch.ones((n_time_sequence + 1, n_time_sequence + 1)))
                             .view(1, 1, n_time_sequence + 1, n_time_sequence + 1))


    def forward(self,query,key,value):
        # T,N*D
        T,D=query.size()
        print("d",D)
        q=query.view(T, self.num_heads, D//self.num_heads).transpose(1,0)
        k = key.view(key.shape[0],  self.num_heads, key.shape[1]// self.num_heads).transpose(1, 0)
        v = value.view(value.shape[0], self.num_heads, value.shape[1] // self.num_heads).transpose(1, 0)

        attention=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        attention=attention.masked_fill(self.mask[:,:T,:T]==0, float('-inf'))
        att=F.softmax(attention,dim=-1)

        y=att@v
        y=y.transpose(1,0).reshape(T,-1)
        return y

    def forward2(self,query,key,value):
        # B, T,N*D
        B, T,D=query.size()
        q=query.view(B, T, self.num_heads, D//self.num_heads).transpose(2,1)
        k = key.view(key.shape[0], key.shape[1], self.num_heads, key.shape[2]// self.num_heads).transpose(2, 1)
        v = value.view(value.shape[0],value.shape[1], self.num_heads, value.shape[2] // self.num_heads).transpose(2, 1)

        attention=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        attention=attention.masked_fill(self.mask2[:,:,T,:T]==0, float('-inf'))
        att=F.softmax(attention,dim=-1)

        y=att@v
        y=y.transpose(2,1).reshape(B,T,-1)
        return y

class MultiHeadAttention2(nn.Module):
    def __init__(self,hidden_dim, num_heads):
        super(MultiHeadAttention2, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        # self.n_time_sequence=n_t
        # self.n=n_agent

    def forward(self,query,key,value):
        # T,K,N,D
        T,K,N,D=query.size()
        q=query.view(T, K,N,self.num_heads, D//self.num_heads).transpose(2,3) # T,K,N,D->T,K,head,N,d
        k = key.view(T, K,N,self.num_heads, D // self.num_heads).transpose(2,3) # T,K,head,N,d
        v = value.view(T, K,N, self.num_heads, value.shape[-1] // self.num_heads).transpose(2,3) # T,K,head,N,d

        attention=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1))) # T,K,head,N,N

        att=F.softmax(attention,dim=-1)

        y=att@v # T,K,head,N,d
        y=y.transpose(2,3).reshape(T,K,N,-1) # T,K,N,head*d

        return y

class MultiHeadAttention3(nn.Module):
    def __init__(self,hidden_dim, num_heads):
        super(MultiHeadAttention3, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads



    def forward(self,query,key,value):
        # T,N*K,D
        T,N,D=query.size()
        q=query.view(T, N, self.num_heads, D//self.num_heads).transpose(1,2)
        k = key.view(1, key.shape[1], self.num_heads, D // self.num_heads).transpose(1, 2)
        v = value.view(1, value.shape[1], self.num_heads, D // self.num_heads).transpose(1, 2)

        attention=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))

        att=F.softmax(attention,dim=-1)

        y=att@v
        y=y.transpose(1,2).reshape(T,N,D)
        return y

    def forward2(self,query,key,value):
        # T,N*K,D
        T,N,D=query.size()
        q=query.view(T, N, self.num_heads, D//self.num_heads).transpose(1,2)
        k = key.view(key.shape[0], key.shape[1], self.num_heads, D // self.num_heads).transpose(1, 2)
        v = value.view(value.shape[0], value.shape[1], self.num_heads, D // self.num_heads).transpose(1, 2)

        attention=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))

        att=F.softmax(attention,dim=-1)

        y=att@v
        y=y.transpose(1,2).reshape(T,N,D)
        return y

class MultiHeadAttention32(nn.Module):
    def __init__(self,hidden_dim, num_heads):
        super(MultiHeadAttention32, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads



    def forward(self,query,key,value):
        # T,N*K,D
        T,N,D=query.size()
        # key=
        q=query.view(T, N, self.num_heads, D//self.num_heads).transpose(1,2)
        k = key.view(T, key.shape[1], self.num_heads, key.shape[2] // self.num_heads).transpose(1, 2)
        v = value.view(T, key.shape[1], self.num_heads, key.shape[2] // self.num_heads).transpose(1, 2)

        attention=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))

        att=F.softmax(attention,dim=-1)

        y=att@v
        y=y.transpose(1,2).reshape(T,N,D)
        return y

