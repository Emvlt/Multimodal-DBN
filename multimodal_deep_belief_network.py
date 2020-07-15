#!/usr/bin/env python
# coding: utf-8

import torch
import time
import convolutionnal_rbm as conv_rbm
import fc_rbm as fc_rbm
import joint_rbm as joint_rbm
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from torch import nn

import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
"""
One want to create a deep-belief bimodal auto encoder.
It has to implement:
    - the unsupervised feature extraction
    - the unsupervised learning of the code high-level features 
"""

class multimod_dbn():
    def __init__(self, name, number_of_modalities):
        super(multimod_dbn, self).__init__()   
        self.name   = name
        self.modalities = {}
        self.inputs     = {}
        self.device = 'cpu'
        for modality in range(number_of_modalities):
            self.modalities[str(modality)] = OrderedDict()
            self.inputs[str(modality)] = OrderedDict()
        self.joint_layer= False
    
    def add_layer(self, name, type, gaussian_units, visible_units, **kwargs):
        if type=='joint':
            rbm = joint_rbm.joint_rbm(name, gaussian_units, visible_units, kwargs['filter_properties'], kwargs['hidden_units']) 
            self.joint_layer = rbm
        else:
            if type=='fully_connected':
                rbm = fc_rbm.fc_rbm(name, gaussian_units,visible_units, kwargs['hidden_units'])
                
            elif type=='convolutional':
                rbm = conv_rbm.conv_rbm(name,gaussian_units, visible_units,
                                        kwargs['f_height'], kwargs['f_width'], kwargs['f_number'], kwargs['c_factor'])    
            self.modalities[str(kwargs['modality'])][name] = rbm
            
    def save_network(self, save_path):
        for modality in self.modalities.values():
            for layer_name, layer in modality.items():
                torch.save(layer.parameters, save_path+layer_name)

        if self.joint_layer:
            torch.save(self.joint_layer.parameters, save_path+self.joint_layer.name)
    
    def move_network_to_device(self, device):
        self.device=device
        for modality in self.modalities.values():
            for layer_name, layer in modality.items():
                layer.to_device(device)

        if self.joint_layer:
            self.joint_layer.to_device(device)
                         
    def initialise_layer(self, layer_name, load_path, **kwargs):
        model = torch.load(load_path)
        if kwargs.get('modality'):
            layer = self.modalities[kwargs['modality']][layer_name]
        else:
            layer = self.joint_layer
        layer.initialisation(model)
        

    def get_input_layer(self, layer_name, modality, input_data, **kwargs):
        output_data = input_data
        for key, value in self.modalities[str(modality)].items():
            if key == layer_name:
                cast_size = self.modalities[str(modality)][key].visible_units.copy()
                cast_size.insert(0, output_data.size()[0])
                return [output_data.view(tuple(cast_size))]
            else:
                output_data = value.bottom_top([output_data])
                
    def get_input_joint_layer(self, inputs, **kwargs):
        input_data = []
        for modality in range(len(self.modalities)):
            if self.modalities[str(modality)]:
                last_layer_name_of_current_mod = next(reversed(self.modalities[str(modality)]))
                input_m = self.get_input_layer(last_layer_name_of_current_mod, modality, inputs[modality])  
                input_data.append(self.modalities[str(modality)][last_layer_name_of_current_mod].bottom_top(input_m))
            else:
                input_data.append(inputs[modality])
        return input_data
 
    def train_layer(self, layer_name, dataloader, data_names, run_name, save_path, epochs, CD_k, learning_rate, momentum, weight_decay, **kwargs):
        '''
        layer_name : string
        data_names : [string, string], ex : ['object','cad']
        '''       
        writer_step = 0
        writer = SummaryWriter(log_dir = run_name)   
        training_observables = {}
        if self.joint_layer and layer_name == self.joint_layer.name:
                layer = self.joint_layer
        else:
            layer = self.modalities[str(kwargs['modality'])][layer_name] 

        for epoch in range(epochs):
            training_observables['Energy'] = 0
            writer_step += 1
            number_of_batches = 0            
            tic = time.time()
            for data in dataloader:
                input_data ={}
                if self.joint_layer and layer_name == self.joint_layer.name:
                    input_data_ = [data[data_names[i]].to(self.device).float() for i in range(len(self.modalities))]
                    input_data_ = self.get_input_joint_layer(input_data_)   
                    for index, d_name in enumerate(data_names):
                        input_data[d_name] = input_data_ 
                else:    
                    input_data_ = data[data_names[kwargs['modality']]].to(self.device).float()                    
                    input_data_ = self.get_input_layer(layer_name, kwargs['modality'], input_data_)
                    input_data[data_names[kwargs['modality']]] = input_data_ 
                with torch.no_grad():
                    observables_dict = self.batch_training(layer, CD_k, learning_rate, momentum, weight_decay, input_data)
                for name, value in observables_dict.items():
                    training_observables[name] += value
                number_of_batches+=1
            tac = time.time()
            print('--------------------------------------------------------')                
            print("Layer : %s, Epoch: %s, Elapsed time : %.2f" % (layer_name, epoch, tac-tic))
            for name, value in training_observables.items():
                writer.add_scalar(name, value/number_of_batches, writer_step)                
                print(name+' : '+str(value/number_of_batches))
        torch.save(layer.parameters, save_path)
        writer.close()     
        
    def batch_training(self, layer, CD_k, learning_rate, momentum, weight_decay, i_d):
        input_data =[]
        for d in i_d.values():
            input_data.append(d[0])
        batch_size = input_data[0].size()[0]
        # Sample layer
        output_data, hidden_states = layer.gibbs_sampling( input_data, CD_k)
        # Compute joint energy
        joint_energy = layer.compute_energy(input_data, output_data, hidden_states, batch_size)
        # Update                    
        layer.update_parameters( learning_rate, momentum, weight_decay, input_data, output_data, hidden_states, batch_size)
        observable_dict = {'Energy':joint_energy}
        return observable_dict

    def network_inference(self, inputs):
        inputs = self.get_input_joint_layer(inputs, inference=True)
        if self.joint_layer:
            inputs = self.joint_layer.gibbs_sampling_( inputs, 1)
        outputs= self.top_bottom(inputs)
        return outputs
   
    def study_code(self, inputs):
        inputs = self.get_input_joint_layer(inputs, inference=True)
        if self.joint_layer:
            p_h = self.joint_layer.get_hidden_probabilities(inputs)
            h = torch.bernoulli(p_h)
        return p_h.detach().cpu(), h.detach().cpu() 
    
    def top_bottom(self, inputs):
        outputs = inputs 
        batch_size = inputs[0].size()[0]
        for modality_name in range(len(self.modalities)):
            for key, value in reversed(self.modalities[str(modality_name)].items()):
                cast_size = self.modalities[str(modality_name)][key].hidden_units.copy()
                cast_size.insert(0, batch_size)
                outputs[modality_name] = outputs[modality_name].view(tuple(cast_size))
                outputs[modality_name] = value.top_bottom(outputs[modality_name]) 
        return outputs
    
    def experience_4(self, inputs):
        ## Change randomly the bits in the code and study the effect on the reconstruction : Error measure against missing code : 10 curves for increasing percentage of missing info
        # inputs  : full modalities
        #           number of lists to return
        # returns : n lists of error against missing frames
        error_list = []
        h_list = []
        frames_to_remove = np.random.choice(256, 125, replace=False)
        corr = inputs[0].clone()
        corr[:,:,:,frames_to_remove]=0
        list_of_indexes_removed   = []
        list_of_indexes_to_remove = [i for i in range(self.joint_layer.hidden_units[0])] 
        
        inputs_ = self.get_input_joint_layer([corr,inputs[1]], inference=True)
        h = self.joint_layer.get_hidden_states(inputs_)
        
        h_list.append(vutils.make_grid(h.view(1,1,h.size()[0],self.joint_layer.hidden_units[0]), padding=2, normalize=False))
        h = h.int().bool()
        n_h = h.clone()
        outs =[]
        for i in range(self.joint_layer.hidden_units[0]):        
            print(i)
            bits_to_change = np.random.choice(list_of_indexes_to_remove,1, replace=False)
            list_of_indexes_to_remove.remove(bits_to_change[0])
            list_of_indexes_removed.append(bits_to_change[0])            
            n_h[:, list_of_indexes_removed] = ~n_h[:,list_of_indexes_removed]
            h_list.append(vutils.make_grid(np.reshape(n_h,(1,1,h.size()[0],self.joint_layer.hidden_units[0])), padding=2))
            outputs = self.joint_layer.get_visible_states(n_h.float())
            outputs = self.top_bottom(outputs)
            error_list.append(torch.norm(inputs[0]-outputs[0])/pow(torch.norm(outputs[0]),2))
            outs.append(vutils.make_grid(np.reshape(outputs[0][0],(1,1,256,256)), padding=2, normalize=True))
        return error_list, h_list, outs 
        