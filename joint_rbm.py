#!/usr/bin/env python
# coding: utf-8

import torchvision
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from collections import OrderedDict

# Intent : implement a fully connected RBM that can be integrated to a DBN framework as the bottleneck of an autoencoder. Works for two modalities. It might be temporary as an architecture that supports n modalities might be made
'''
Implements
    - state functions : given a layer and its type (gaussian/binary), compute its state
    - Energy functions : given a configuration (single layer or RBM joint configuration), compute the energy related to the configuration
    - CD related functions : sample the distribution defined by the RBM, update the states with normal or sparse update rule
    - DBN functions, called when the RBM is part of a DBN 
    - training function, perform training on a RBM when it's not part of an DBN
'''
class joint_rbm(nn.Module):
    def __init__(self, name, gaussian_units, visible_units, filters_properties, hidden_units):
        '''
        Parameters :
            name : the name of the RBM, useful mainly for the DBN
            size m : size of visible convolutional layer of the m-th modality , list of integers 
            hidden units : the number of hidden units, positive integer
        '''
        super(joint_rbm, self).__init__()
        self.name         = name   
        self.hidden_units    = hidden_units
        self.gaussian_units = gaussian_units
        self.type = 'joint'
        self.parameters   = {
            'h_bias'   : nn.Parameter((torch.ones(hidden_units[0])*-4)), 
            'h_bias_m' : nn.Parameter(torch.zeros(hidden_units[0]))
        }
        self.n_modalities = len(visible_units)
        self.device = 'cpu'
        self.visible_units = visible_units
        # hidden layer
        for parameter in range(self.n_modalities):
            self.parameters[str(parameter)] = OrderedDict()
            
        # Parameters modalities
        for index in range(self.n_modalities):
            self.parameters[str(index)]['weights_m'] = nn.Parameter((torch.zeros(filters_properties[index][0], visible_units[index][0], filters_properties[index][1], filters_properties[index][2] )))
            self.parameters[str(index)]['v_bias_m']  =nn.Parameter(torch.zeros(visible_units[index][0]))
            self.parameters[str(index)]['weights'] = nn.Parameter((torch.randn(filters_properties[index][0], visible_units[index][0], filters_properties[index][1], filters_properties[index][2] )*0.01))
            self.parameters[str(index)]['v_bias']  =nn.Parameter(torch.ones(visible_units[index][0])*0.01)
            
    def to_device(self, device):
        self.device = device
        self.parameters['h_bias']   = self.parameters['h_bias'].to(device)
        self.parameters['h_bias_m'] = self.parameters['h_bias_m'].to(device)
        for index in range(self.n_modalities):
            self.parameters[str(index)]['weights'] = self.parameters[str(index)]['weights'].to(device)
            self.parameters[str(index)]['v_bias']  = self.parameters[str(index)]['v_bias'].to(device)
            self.parameters[str(index)]['weights_m'] = self.parameters[str(index)]['weights_m'].to(device)
            self.parameters[str(index)]['v_bias_m']  = self.parameters[str(index)]['v_bias_m'].to(device)
            
    def initialisation(self, model):
        self.parameters['h_bias']   = nn.Parameter(model['h_bias'])
        self.parameters['h_bias_m'] = nn.Parameter(model['h_bias_m'])
        for index in range(self.n_modalities):
            self.parameters[str(index)]['weights'] = nn.Parameter(model[str(index)]['weights'])
            self.parameters[str(index)]['v_bias']  = nn.Parameter(model[str(index)]['v_bias'])
            self.parameters[str(index)]['weights_m'] = nn.Parameter(model[str(index)]['weights_m'])
            self.parameters[str(index)]['v_bias_m']  = nn.Parameter(model[str(index)]['v_bias_m'])
            
    ######################## Get states functions ########################
    def get_visible_states(self, h):
        modalities = []
        for modality in range(self.n_modalities):
            #Wh = F.linear(h, self.parameters[str(modality)]['weights'],self.parameters[str(modality)]["v_bias"])   
            Wh = F.conv_transpose2d(h, self.parameters[str(modality)]['weights'], bias = self.parameters[str(modality)]['v_bias'])
            if self.gaussian_units[modality]:
                if self.gaussian_units:
                    sample_v = torch.normal(Wh, 1)
                else:
                    p_v = F.sigmoid(Wh)
                    sample_v = torch.bernoulli(p_v)
            modalities.append(sample_v)
        return modalities

    def get_hidden_states(self, modalities):
        h_input = torch.zeros((modalities[0].size()[0],self.hidden_units[0], self.hidden_units[1], self.hidden_units[2])).to(self.device)
        for index, modality in enumerate(modalities): 
            h_input += F.conv2d(modality, self.parameters[str(index)]['weights'])
            #h_input += F.linear(modality, self.parameters[str(index)]['weights'].t(), bias=False)   
        p_h  = F.sigmoid(h_input) 
        sample_h = torch.bernoulli(p_h)
        return sample_h
    
    def get_hidden_probabilities(self, modalities):
        h_input = torch.zeros((modalities[0].size()[0],self.hidden_units[0], self.hidden_units[1], self.hidden_units[2])).to(self.device)
        for index, modality in enumerate(modalities): 
            h_input += F.conv2d(modality, self.parameters[str(index)]['weights'])
            #h_input += F.linear(modality, self.parameters[str(index)]['weights'].t(), bias=False)   
        p_h  = F.sigmoid(h_input) 
        return p_h
    
    '''def get_hidden_probabilities(self, modalities):
        h_input = torch.zeros((modalities[0].size()[0], self.hidden_units[0])).to(self.device)
        for index, modality in enumerate(modalities):
            h_input += F.linear(modality, self.parameters[str(index)]['weights'].t())   
        p_h  = F.sigmoid(torch.add(h_input,self.parameters['h_bias']))
        return p_h'''
    
    ################################## Energy functions ##################################      
    '''def compute_energy(self, input_data, output_data, hidden_states, batch_size):
        energy = (hidden_states['h0']-hidden_states['hk'])*self.parameters['h_bias'].sum()
        for index in range(self.n_modalities):
            visible_detection = hidden_states['h0']*F.linear(input_data[index], self.parameters[str(index)]["weights"].t()) - hidden_states['hk']*F.linear(output_data[index], self.parameters[str(index)]["weights"].t())     
            if self.gaussian_units[index]:
                visible = (pow(input_data[index]-self.parameters[str(index)]['v_bias'],2) - pow(output_data[index]-self.parameters[str(index)]['v_bias'],2))/2
            else:                
                visible = self.parameters[str(index)]["v_bias"]*input_data[index] - self.parameters[str(index)]["v_bias"]*output_data[index]
            energy += visible_detection.sum() + visible.sum()
        return (energy.sum()/batch_size).to('cpu')'''
    
    def compute_energy(self, input_data, output_data, hidden_states, batch_size):
        energy = torch.sum(hidden_states['h0']-hidden_states['hk'],(0,2,3))*self.parameters['h_bias']
        for index in range(self.n_modalities):
            visible_detection = F.conv2d(input_data[index], self.parameters[str(index)]['weights'])*hidden_states['h0'] - F.conv2d(output_data[index], self.parameters[str(index)]['weights'])*hidden_states['hk']        
            if self.gaussian_units[index]:
                visible = (pow(torch.sum(input_data[index],(0,2,3))-self.parameters[str(index)]['v_bias'],2) - pow(torch.sum(output_data[index],(0,2,3))-self.parameters[str(index)]['v_bias'],2))/2
            else:                
                visible = torch.sum(input_data[index]-output_data[index],(0,2,3))*self.parameters[str(index)]['v_bias']
            energy += visible_detection.sum() + visible.sum()
        return (energy.sum()/batch_size).to('cpu')
    
    ################################## Update functions ##################################  
    '''def get_weight_gradient(self, hidden_vector, visible_vector, batch_size):
        return torch.mm(hidden_vector.t(),visible_vector).t().sum(0)/batch_size
    
    def get_bias_gradient(self,  vector_0, vector_k, batch_size):
        return torch.add(vector_0, -vector_k).sum(0)/batch_size'''

    def get_weight_gradient(self, hidden_vector, visible_vector, batch_size):
        return torch.transpose(F.conv2d(torch.transpose(visible_vector,1,0), torch.transpose(hidden_vector,1,0)),1,0).sum(0)/batch_size
    
    def get_v_bias_gradient(self, vector_0, vector_k, batch_size, index):
        return torch.add(vector_0, -vector_k).sum([0,2,3])/(batch_size*self.visible_units[index][1]*self.visible_units[index][2])
        
    def get_h_bias_gradient(self,  vector_0, vector_k, batch_size):
        return torch.add(vector_0, -vector_k).sum([0,2,3])/(batch_size*self.hidden_units[1]*self.hidden_units[2])
    
    
    def gibbs_sampling(self, input_modalities, k):
        h0 = self.get_hidden_states(input_modalities)  
        hk = h0
        with torch.no_grad():
            for _ in range(k):
                modalities_k = self.get_visible_states(hk)
                hk = self.get_hidden_states(modalities_k) 
        hidden_states ={'h0':h0,'hk':hk}
        return modalities_k, hidden_states
                
    def gibbs_sampling_(self, input_modalities, k):
        h0 = self.get_hidden_states(input_modalities)  
        hk = h0
        with torch.no_grad():
            for _ in range(k):
                modalities_k = self.get_visible_states(hk)
                hk = self.get_hidden_states(modalities_k) 
        hidden_states ={'h0':h0,'hk':hk}
        return modalities_k
    
    def update_parameters(self, lr, momentum, weight_decay, input_data, output_data, hidden_states, batch_size):
        with torch.no_grad():
            d_h = self.get_h_bias_gradient(hidden_states['h0'], hidden_states['hk'], batch_size)  
            #d_h = self.get_bias_gradient(hidden_states['h0'], hidden_states['hk'], batch_size)  
            self.parameters['h_bias_m']  = torch.add(momentum*self.parameters['h_bias_m'], d_h)
            self.parameters['h_bias']   += lr*torch.add(d_h,  self.parameters['h_bias_m'])
            for index in range(self.n_modalities):
                d_v = self.get_v_bias_gradient(input_data[index],output_data[index], batch_size, index)
                #d_v = self.get_bias_gradient(input_data[index],output_data[index], batch_size, index)
                dw_in  = self.get_weight_gradient(hidden_states['h0'], input_data[index], batch_size)
                dw_out = self.get_weight_gradient(hidden_states['hk'], output_data[index], batch_size)
                d_w  = torch.add(dw_in,-dw_out)
                self.parameters[str(index)]['weights_m'] = torch.add(momentum* self.parameters[str(index)]['weights_m'], d_w)
                self.parameters[str(index)]['v_bias_m']  = torch.add(momentum* self.parameters[str(index)]['v_bias_m'], d_v)
                self.parameters[str(index)]['weights']  += lr*(torch.add(d_w, self.parameters[str(index)]['weights_m']))+weight_decay*self.parameters[str(index)]['weights']
                self.parameters[str(index)]['v_bias']   += lr*torch.add(d_v, self.parameters[str(index)]['v_bias_m'])

            