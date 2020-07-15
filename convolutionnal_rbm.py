#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import matplotlib.pyplot as plt

# Intent : implement a convolutional RBM that can be integrated to a DBN framework
'''
Implements
    - Inputs functions : given a layer, compute the input received by another layer.
    - Probability functions : given an input, compute the probability of activation of each unit in a given layer
    - state functions : given a layer and its type (gaussian/binary), compute its state
    - Energy functions : given a configuration (single layer or RBM joint configuration), compute the energy related to the configuration
    - CD related functions : sample the distribution defined by the RBM, update the states with normal or sparse update rule
    - DBN functions, called when the RBM is part of a DBN 
'''
class conv_rbm(nn.Module):
    
    def __init__(self, name, gaussian_units, visible_units, f_height, f_width, f_number, c_factor):
        '''
        Parameters :
            args
                name : the name of the RBM, useful mainly for the DBN
                v_channels : the number of channels of the visible layer, positive integer
                f_height : the filters's height, positive integer
                f_width  : the filters's width, positive integer
                f_number : the number of filters of the RBM, positive integer
                c_factor : the pooling factor, positive integer
                gaussian_units    : the state of the visible units, bool           
        '''
        super(conv_rbm, self).__init__()
        self.name            = name
        self.visible_units   = visible_units
        self.v_channels      = visible_units[0]
        self.f_height        = f_height
        self.f_width         = f_width
        self.f_number        = f_number
        self.c_factor        = c_factor
        n_size = int((visible_units[1]-f_height)/c_factor +1)
        self.hidden_units    = [f_number, n_size, n_size]
        self.gaussian_units  = gaussian_units
        self.n_modalities    = 1
        self.type = 'convolutional'
        self.device = 'cpu'
        self.parameters   = {
            'h_bias'   : nn.Parameter((torch.ones(f_number)*-4)), 
            'h_bias_m' : nn.Parameter(torch.zeros(f_number)),
            'v_bias'  : nn.Parameter((torch.ones(self.v_channels)*0.01)),
            'weights' : nn.Parameter((torch.randn(f_number, self.v_channels, f_height, f_width )*0.01)),
            'v_bias_m'  : nn.Parameter(torch.zeros(self.v_channels)),
            'weights_m' : nn.Parameter(torch.zeros(f_number, self.v_channels, f_height,f_width))
        }
        '''self.parameters   = {
            'h_bias'   : nn.Parameter((torch.ones(f_number)*-4)), 
            'h_bias_m' : nn.Parameter(torch.zeros(f_number)),
            'v_bias'  : nn.Parameter((torch.ones(self.v_channels)*0.1)),
            'weights' : nn.Parameter((torch.randn(f_number, self.v_channels, f_height, f_width )*0.01)),
            'v_bias_m'  : nn.Parameter(torch.zeros(self.v_channels)),
            'weights_m' : nn.Parameter(torch.zeros(f_number, self.v_channels, f_height,f_width))
        }'''
        self.hidden_units 
    ################################## GPU related functions ##################################   
    def to_device(self, device):
        self.device = device
        self.parameters['h_bias']   = self.parameters['h_bias'].to(device)
        self.parameters['h_bias_m'] = self.parameters['h_bias_m'].to(device)
        self.parameters['weights']   = self.parameters['weights'].to(device)
        self.parameters['v_bias']    = self.parameters['v_bias'].to(device)
        self.parameters['weights_m'] = self.parameters['weights_m'].to(device)
        self.parameters['v_bias_m']  = self.parameters['v_bias_m'].to(device)
        
    def initialisation(self,model):
        self.parameters['h_bias']   = nn.Parameter(model['h_bias'])
        self.parameters['h_bias_m'] = nn.Parameter(model['h_bias_m'])
        self.parameters['weights'] = nn.Parameter(model['weights'])
        self.parameters['v_bias']  = nn.Parameter(model['v_bias'])
        self.parameters['weights_m'] = nn.Parameter(model['weights_m'])
        self.parameters['v_bias_m']  = nn.Parameter(model['v_bias_m'])
        
    def get_hidden_probability(self, v):
        v = v[0]
        p_h = F.sigmoid(F.conv2d(v, self.parameters['weights'], self.parameters['h_bias'], stride = self.c_factor))
        return  [p_h]
    
    def get_visible_probability(self, h):
        input_visible = F.conv_transpose2d(h, self.parameters['weights'], bias = self.parameters['v_bias'], stride = self.c_factor)
        if self.gaussian_units:
            p_v = torch.normal(input_visible,1)
        else:
            p_v = torch.sigmoid(input_visible)
        return [p_v]
        
    def get_hidden_states(self, v):
        v = v[0]
        p_h = F.sigmoid(F.conv2d(v, self.parameters['weights'], self.parameters['h_bias'], stride = self.c_factor))
        h = torch.bernoulli(p_h)
        return [h]
        
    def get_visible_states(self, h):
        input_visible = F.conv_transpose2d(h, self.parameters['weights'], bias = self.parameters['v_bias'], stride = self.c_factor)
        if self.gaussian_units:
            v = torch.normal(input_visible,1)
        else:
            p_v = torch.sigmoid(input_visible)
            v = torch.bernoulli(p_v)
        return [v]
                                  
    def compute_energy(self, input_data, output_data, hidden_states, batch_size):
        # detection bias energy term
        detection = torch.sum(hidden_states['h0']-hidden_states['hk'],(0,2,3))*self.parameters['h_bias']
        # joint visible detection energy
        visible_detection = F.conv2d(input_data[0], self.parameters['weights'], stride = self.c_factor)*hidden_states['h0'] - F.conv2d(output_data[0], self.parameters['weights'], stride = self.c_factor)*hidden_states['hk']    
        # visible bias energy term
        if self.gaussian_units:
            visible = (pow(torch.sum(input_data[0],(0,2,3))-self.parameters['v_bias'],2) - pow(torch.sum(output_data[0],(0,2,3))-self.parameters['v_bias'],2))/2
        else:
            visible = torch.sum(input_data[0]-output_data[0],(0,2,3))*self.parameters['v_bias']
            
        return (-(visible.sum() + detection.sum()  + visible_detection.sum() )/batch_size).to('cpu')
    
    ################################## CD (Gibbs sampling + update) functions ##################################    
    def get_weight_gradient(self, hidden_vector, visible_vector, batch_size):
        return torch.transpose(F.conv2d(torch.transpose(visible_vector,1,0), torch.transpose(hidden_vector,1,0), dilation = self.c_factor),1,0).sum(0)/batch_size
    
    def get_v_bias_gradient(self, vector_0, vector_k, batch_size):
        return torch.add(vector_0, -vector_k).sum([0,2,3])/(batch_size*self.visible_units[1]*self.visible_units[2])
        
    def get_h_bias_gradient(self,  vector_0, vector_k, batch_size):
        return torch.add(vector_0, -vector_k).sum([0,2,3])/(batch_size*self.hidden_units[1]*self.hidden_units[2])
    
    ################################## DBN related functions ##################################   
    def bottom_top(self, input_data):
        #[h] = self.get_hidden_states(input_data)
        [p_h] = self.get_hidden_probability(input_data)
        p_h = (p_h-p_h.mean())/p_h.std()
        return p_h
    
    def top_bottom(self, h):
        #v = self.get_visible_states(h)[0]
        v = F.conv_transpose2d(h, self.parameters['weights'], bias = self.parameters['v_bias'], stride = self.c_factor)
        #v = torch.sigmoid(v)
        return v
        
    ################################## Sampling functions #################################
    def gibbs_sampling(self, input_modalities, k):
        [h0] = self.get_hidden_states(input_modalities)  
        hk = h0
        for _ in range(k):
            modalities_k = self.get_visible_states(hk)
            [hk] = self.get_hidden_states(modalities_k) 
        hidden_states ={'h0':h0,'hk':hk}
        return modalities_k, hidden_states
        
    def update_parameters(self, lr, momentum, weight_decay, input_data, output_data, hidden_states, batch_size):
        d_v = self.get_v_bias_gradient(input_data[0],output_data[0], batch_size)
        d_h = self.get_h_bias_gradient(hidden_states['h0'], hidden_states['hk'], batch_size)   
        dw_in  = self.get_weight_gradient(hidden_states['h0'], input_data[0], batch_size)
        dw_out = self.get_weight_gradient(hidden_states['hk'], output_data[0], batch_size)
        d_w  = torch.add(dw_in,-dw_out)
        self.parameters['weights_m'] = torch.add(momentum* self.parameters['weights_m'], d_w)
        self.parameters['v_bias_m']  = torch.add(momentum* self.parameters['v_bias_m'], d_v)
        self.parameters['weights']  += lr*(torch.add(d_w, self.parameters['weights_m']))+weight_decay*self.parameters['weights']
        self.parameters['v_bias']   += lr*torch.add(d_v, self.parameters['v_bias_m'])
        self.parameters['h_bias_m']  = torch.add(momentum*self.parameters['h_bias_m'], d_h)
        self.parameters['h_bias']   += lr*torch.add(d_h,  self.parameters['h_bias_m'])