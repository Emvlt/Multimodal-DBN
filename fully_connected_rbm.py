#!/usr/bin/env python
# coding: utf-8

import torchvision
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import pickle as pkl
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

# Intent : implement a fully connected RBM that can be integrated to a DBN framework as the bottleneck of an autoencoder. Works for one modality. It might be temporary as an architecture that supports n modalities might be made
'''
Implements
    - state functions : given a layer and its type (gaussian/binary), compute its state
    - Energy functions : given a configuration (single layer or RBM joint configuration), compute the energy related to the configuration
    - CD related functions : sample the distribution defined by the RBM, update the states with normal or sparse update rule
    - DBN functions, called when the RBM is part of a DBN 
    - training function, perform training on a RBM when it's not part of an DBN
'''
class fc_rbm(nn.Module):
    def __init__(self, name, gaussian_units, visible_units, hidden_units):
        '''
        Parameters :
            name : the name of the RBM, useful mainly for the DBN
            visible_units : the number of visible units, list of integers if convolutionnal, integer otherwise
            hidden units : the number of hidden units, positive integer
            from_scratch : to decide wether the rbm is initialised from scratch of from an already saved model, bool
            load_path : path from which model has to be loaded, matters only if from_scratch is True, string
            save_path : path to which the model has to be saved
            run_name : the path of the writer
        '''
        super(fc_rbm, self).__init__()
        self.name          = name
        self.visible_units = [visible_units]
        self.hidden_units  = [hidden_units ]      
        self.gaussian_units    = gaussian_units
        self.n_modalities    = 1
        self.type = 'fully_connected'
        self.parameters   = {
            'h_bias' : nn.Parameter((torch.ones(hidden_units)*-0.004)), 
            'h_bias_m' : nn.Parameter(torch.zeros(hidden_units)),
            'weights_m' : nn.Parameter(torch.zeros(visible_units, hidden_units)),
            'v_bias_m'  : nn.Parameter(torch.zeros(visible_units)),
            'weights' : nn.Parameter((torch.randn(visible_units, hidden_units)*0.001)),
            'v_bias'  : nn.Parameter((torch.ones(visible_units)*0.001))
        }
        self.device = 'cpu'

    def to_device(self, device):
        self.device = device
        self.parameters['h_bias']   = self.parameters['h_bias'].to(device)
        self.parameters['h_bias_m'] = self.parameters['h_bias_m'].to(device)
        self.parameters['weights']   = self.parameters['weights'].to(device)
        self.parameters['v_bias']    = self.parameters['v_bias'].to(device)
        self.parameters['weights_m'] = self.parameters['weights_m'].to(device)
        self.parameters['v_bias_m']  = self.parameters['v_bias_m'].to(device)
        
    def initialisation(self, model):
        self.parameters['h_bias']   = nn.Parameter(model['h_bias'])
        self.parameters['h_bias_m'] = nn.Parameter(model['h_bias_m'])
        self.parameters['weights'] = nn.Parameter(model['weights'])
        self.parameters['v_bias']  = nn.Parameter(model['v_bias'])
        self.parameters['weights_m'] = nn.Parameter(model['weights_m'])
        self.parameters['v_bias_m']  = nn.Parameter(model['v_bias_m'])
        
    ######################## state functions ########################
    def get_visible_states(self, h):       
        # Computes probability of visible layer states and corresponding sampled states
        if self.gaussian_units:
            mean = F.linear(h, self.parameters["weights"],self.parameters["v_bias"]) 
            sample_v = torch.normal(mean, 1)
        else:                
            Wh = F.linear(h, self.parameters["weights"],self.parameters["v_bias"])        
            p_v = F.sigmoid(Wh)
            sample_v = torch.bernoulli(p_v)
        return [sample_v]
    
    def get_hidden_states(self, v):
         # Computes probability of hidden layer states and corresponding sampled states
        v = v[0]
        vW = F.linear(v, self.parameters["weights"].t(), self.parameters['h_bias'])
        p_h = F.sigmoid(vW)
        sample_h = torch.bernoulli(p_h)
        return sample_h
    
    def get_hidden_probability(self, v):
        v = v[0]
        p_h = F.sigmoid(F.linear(v, self.parameters["weights"].t(), self.parameters['h_bias']))
        return p_h
    
    def get_visible_probability(self, h):
        Wh = F.linear(h, self.parameters["weights"],self.parameters["v_bias"])        
        p_v = F.sigmoid(Wh)
        return p_v
    ###################### Energy function ###########################   
    def compute_energy(self, input_data, output_data, hidden_states, batch_size):
        # detection bias energy term
        energy = (-(hidden_states['h0']-hidden_states['hk'])*self.parameters["h_bias"]).sum((0,1))
        visible_detection = hidden_states['h0']*F.linear(input_data[0], self.parameters["weights"].t()) - hidden_states['hk']*F.linear(output_data[0], self.parameters["weights"].t())
        if self.gaussian_units:
            visible = (pow(input_data[0]-self.parameters['v_bias'],2) - pow(output_data[0]-self.parameters['v_bias'],2))/2
        else:
            visible = (input_data[0]-output_data[0])*self.parameters["v_bias"]
        energy -= visible_detection.sum((0,1)) + visible.sum((0,1))
        return (energy/batch_size).to('cpu')
    
   
    ###################### CD (Gibbs sampling + update) functions #################     
    def get_weight_gradient(self, hidden_vector, visible_vector, batch_size):
        return torch.mm(hidden_vector.t(),visible_vector).t().sum(0)/batch_size
    
    def get_bias_gradient(self,  vector_0, vector_k, batch_size):
        return torch.add(vector_0, -vector_k).sum(0)/batch_size
    
    ################################## DBN related functions ##################################   

    def bottom_top(self, input):
        # evaluates the RBM with a bottom input  
        #sample_h = self.get_hidden_states(input)
        p_h = self.get_hidden_probability(input)
        p_h = (p_h-p_h.mean())/p_h.std()
        return p_h
    
    def top_bottom(self, *args):
        # evaluates the RBM with a top input
        input = args[0]
        #input = input.view(input.size(0), -1)
        #v = self.get_visible_states(input)
        v = F.linear(input, self.parameters["weights"],self.parameters["v_bias"])        
        #v = F.sigmoid(v)
        return v[0]
    
    def gibbs_sampling(self, input_modalities, k):
        h0 = self.get_hidden_states(input_modalities)  
        hk = h0
        with torch.no_grad():
            for _ in range(k):
                modalities_k = self.get_visible_states(hk)
                hk = self.get_hidden_states(modalities_k) 
        hidden_states ={'h0':h0,'hk':hk}
        return modalities_k, hidden_states

    def update_parameters(self, lr, momentum, weight_decay, input_data, output_data, hidden_states, batch_size):
        d_v = self.get_bias_gradient(input_data[0],output_data[0], batch_size)
        d_h = self.get_bias_gradient(hidden_states['h0'], hidden_states['hk'], batch_size)            
        dw_in  = self.get_weight_gradient(hidden_states['h0'], input_data[0], batch_size)
        dw_out = self.get_weight_gradient(hidden_states['hk'], output_data[0], batch_size)
        d_w  = torch.add(dw_in,-dw_out)
        with torch.no_grad():
            self.parameters['weights_m'] = torch.add(momentum* self.parameters['weights_m'], d_w)
            self.parameters['v_bias_m']  = torch.add(momentum* self.parameters['v_bias_m'], d_v)
            self.parameters['weights']  += lr*(torch.add(d_w, self.parameters['weights_m']))+weight_decay*self.parameters['weights']
            self.parameters['v_bias']   += lr*torch.add(d_v, self.parameters['v_bias_m'])
            self.parameters['h_bias_m']  = torch.add(momentum*self.parameters['h_bias_m'], d_h)
            self.parameters['h_bias']   += lr*torch.add(d_h,  self.parameters['h_bias_m'])