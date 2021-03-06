#!/usr/bin/env python
# coding: utf-8

"""
Implementation of https://icml.cc/2011/papers/399_icmlpaper.pdf by Emilien Valat
"""

import torch
import time
import convolutional_rbm as conv_rbm
import fully_connected_rbm as fc_rbm
import joint_rbm as joint_rbm
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from torch import nn

import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

class multimod_dbn():
    def __init__(self, name, number_of_modalities):
        """
        Description:
            The method to call to create the muldimodal dbn object
        Explanation:
            The multimodal DBN is holding an OrderedDict for each modality, OrderedDict of which items are the RBM composing each modality.
            A joint-layer, if present, is not in an OrderedDict.
            It is created in the CPU but can be moved to the GPU
        Arguments :
            int modalities : the number of modalities
        """
        super(multimod_dbn, self).__init__()
        self.modalities = {}
        self.device = 'cpu'
        for modality in range(number_of_modalities):
            self.modalities[str(modality)] = OrderedDict()
        self.joint_layer= False

    ################################## Layer management method ##################################

    def add_layer(self, name, type_of_layer, gaussian_units, visible_units, **kwargs):
        """
        Description:
            The method to call when adding a layer.
        Explanation:
            Regarding the type of layer (joint, fully_connected or convolutional) and the modality, adds the layer to the right modality.
        Arguments:
            string name : name of the RBM to be added to the modality
            string type : type of the RBM to be added to the modality
            bool gaussian_units : bool to specify the type of the visible units of the RBM to be added to the modality
            list visible_units  : list representing the size in each dimension of the visible units  of the RBM to be added to the modality
        kwargs:
            list "filters_properties" : 
            list "hidden_units" : list of ints giving the expected hidden layer size
        Comment :
            The layers are meant to be added in order
        """
        if type_of_layer=='joint_convolutional':
            rbm = joint_rbm.joint_rbm(name=name, type_of_layer='convolutional', gaussian_units=gaussian_units, visible_units=visible_units, filters_properties=kwargs['filters_properties'])
            self.joint_layer = rbm
        elif type_of_layer=='joint_fully_connected':
            rbm = joint_rbm.joint_rbm(name=name, type_of_layer='fully_connected', gaussian_units=gaussian_units, visible_units=visible_units, hidden_units=kwargs['hidden_units'])
            self.joint_layer = rbm
        else:
            if name in self.modalities[str(kwargs['modality'])]:
                raise Exception("The name is already taken")
            else:
                if type_of_layer=='fully_connected':
                    rbm = fc_rbm.fc_rbm(name, gaussian_units,visible_units, kwargs['hidden_units'])

                elif type_of_layer=='convolutional':
                    rbm = conv_rbm.conv_rbm(name,gaussian_units, visible_units,
                                            kwargs['filters_properties'])
                self.modalities[str(kwargs['modality'])][name] = rbm

    ################################## Save method ##################################

    def save_network(self, save_path):
        """
        Description:
            The method to call when saving the network
        Explanation:
            Iterates through the OrderedDict of each modality (and the joint layer, if present) and calls the torch.save method.
            Each layer will be saved to the
        Arguments:
            string save_path : the path of the existing folder to save the network in
        """
        for modality_name, modality in self.modalities.items():
            for layer_name, layer in modality.items():
                torch.save(layer.parameters, save_path+'modality_'+modality_name+'_'+layer_name)

        if self.joint_layer:
            torch.save(self.joint_layer.parameters, save_path+self.joint_layer.name)

    ################################## CPU/GPU related methods ##################################

    def move_network_to_device(self, device):
        """
        Description:
            The method to call when moving the network accross the devices
        Explanation:
            Iterates through the OrderedDict of each modality (and the joint layer, if present) and calls the to_device method.
        Arguments:
            string device : device name to move the network to
        """
        self.device=device
        for modality in self.modalities.values():
            for layer_name, layer in modality.items():
                layer.to_device(device)

        if self.joint_layer:
            self.joint_layer.to_device(device)

    ################################## initialisation method ##################################

    def initialise_layer(self, layer_name, load_path, **kwargs):
        """
        Description:
            The method to call when loading an existing layer save to a new DBN
        Explanation:
            Given the name of a layer and the path to the file to be loaded, initialise the layer parameters with the parameters to be restored
        Arguments:
            string layer_name: name of the layer to initialise
            string load_path: string to the path to the saved RBM Parameters
        kwargs:
            int modality : label of the modality
        """
        model = torch.load(load_path)
        if kwargs.get('modality'):
            layer = self.modalities[kwargs['modality']][layer_name]
        else:
            layer = self.joint_layer
        layer.initialisation(model)

    ################################## Input getting methods ##################################

    def get_input_layer(self, layer_name, modality, input_data):
        """
        Description:
            The method to call to infer the input data to a given layer
        Explanation:
            Iterates through the OrderedDict of a given modality until it reaches the desired layer
        Arguments:
            string layer_name: name of the layer of which to get input
            int modality: label of the modality in which the layer is
            tensor input_data: data to infer until the visible units of the desired layer
        """
        output_data = input_data
        for key, value in self.modalities[str(modality)].items():
            cast_size = self.modalities[str(modality)][key].visible_units.copy()
            cast_size.insert(0, output_data.size()[0])
            if key == layer_name:
                return output_data.view(tuple(cast_size))
            else:
                output_data = value.bottom_top(output_data.view(tuple(cast_size)))

    def get_input_joint_layer(self, inputs):
        """
        Description:
            The method to call to infer input data to the joint layer.
        Explanation:
            Iterates through the modalities and calls bottom_top on the last layer of each modality.
        Arguments:
            list of tensors inputs: a list of tensors of size number_of_modalities
        """
        input_data = []
        batch_size = inputs[0].size()[0]
        for modality in range(len(self.modalities)):
            cast_size = self.joint_layer.visible_units[modality].copy()
            cast_size.insert(0, batch_size)
            if self.modalities[str(modality)]:
                last_layer_name_of_current_mod = next(reversed(self.modalities[str(modality)]))
                input_m = self.get_input_layer(last_layer_name_of_current_mod, modality, inputs[modality])
                input_data.append(self.modalities[str(modality)][last_layer_name_of_current_mod].bottom_top(input_m).view(tuple(cast_size)))
            else:
                input_data.append(inputs[modality].view(tuple(cast_size)))
        return input_data

    ################################## Train methods ##################################

    def train_layer(self, layer_name, dataloader, data_names, run_record, save_path, epochs, CD_k, learning_rate, momentum, weight_decay, **kwargs):
        """
        Description:
            Method to call to train a layer. As the DBN implements the Greedy-Layer Wise training, the method has to be called for each layer that has to be trained.
        Explanation:
            Given a layer name, infers the visible units states of the layer and trains it unsupervisedly.
        Arguments:
            string layer_name: name of the layer to be trained
            pytorch dataloader dataloader: dataloader format of pytorch
            list of string data_names: a list containing the input data names
            string run_name: the run name for displaying statistics for the writer
            string save_path: the path to save the trained parameters to
            int epochs: the number of iterations through the database
            int CD_k: the number of gibbs sample to run to run for the training
            float learning_rate: the learning rate of the training
            float momentum: the momentum of the training
            float weight_decay: the weight decay of the training
        kwargs:
            int modality: modality of the layer to be trained if it is not the joint layer
        """
        if run_record:
            writer_step = 0
            writer = SummaryWriter(log_dir = kwargs['run_name'])
        training_observables = {}
        if self.joint_layer and layer_name == self.joint_layer.name:
                layer = self.joint_layer
        else:
            layer = self.modalities[str(kwargs['modality'])][layer_name]
        
        energy_gradient = 0
        
        for epoch in range(epochs):
            training_observables['Energy'] = 0
            if run_record:
                writer_step += 1
            number_of_batches = 0
            tic = time.time()
            for data in dataloader:
                if self.joint_layer and layer_name == self.joint_layer.name:
                    input_data_ = [data[data_names[i]].to(self.device).float() for i in range(len(self.modalities))]
                    input_data = self.get_input_joint_layer(input_data_)
            
                else:
                    input_data_ = data[data_names[kwargs['modality']]].to(self.device).float()
                    input_data = self.get_input_layer(layer_name, kwargs['modality'], input_data_)
                batch_size = data[data_names[0]].size()[0]
                with torch.no_grad():
                    observables_dict = self.batch_training(layer, CD_k, learning_rate, momentum, weight_decay, input_data, batch_size)
                for name, value in observables_dict.items():
                    training_observables[name] += value
                number_of_batches+=1
            tac = time.time()
            print('--------------------------------------------------------')
            print("Layer : %s, Epoch: %s, Elapsed time : %.2f" % (layer_name, epoch, tac-tic))
            for name, value in training_observables.items():
                if run_record:
                    writer.add_scalar(name, value/number_of_batches, writer_step)
                print(name+' : '+str(value/number_of_batches))
                print('Energy gradient : ', energy_gradient-(value/number_of_batches))
                energy_gradient = (value/number_of_batches)
        torch.save(layer.parameters, save_path)
        if run_record:
            writer.close()

    def batch_training(self, layer, CD_k, learning_rate, momentum, weight_decay, input_data, batch_size):
        """
        Description:
            Method to call to update the parameters of a layer given a batch of input data
        Explanation:
            Inputs the visible units of a layer clamped with input data
        Arguments:
            object layer: layer to be trained
            int CD_k: the number of gibbs sample to run to run for the training
            float learning_rate: the learning rate of the training
            float momentum: the momentum of the training
            float weight_decay: the weight decay of the training
            dictionnary i_d : an dictionary of which the keys are the modality names and of which the values are the input data of each modality
        """
        # Sample layer
        output_data, hidden_states = layer.gibbs_sampling( input_data, CD_k)
        # Compute joint energy
        energy = layer.compute_energy(input_data, output_data, hidden_states, batch_size)
        # Update
        layer.update_parameters( learning_rate, momentum, weight_decay, input_data, output_data, hidden_states, batch_size)
        observable_dict = {'Energy': energy}
        return observable_dict

    ################################## Inference methods ##################################

    def top_bottom(self, inputs):
        """
        Description:
            The method to call to infer the code (the hidden units of the top-most layer of each modality) back to the visible units of the first layers of each modality
        Explanation:
            Iterates through the modalities and calls top_bottom for each layer, back to the first one.
        Arguments:
            list of tensors inputs: list of tensors of the input data for each modality
        """
        outputs = inputs
        batch_size = inputs[0].size()[0]
        for modality_index in range(len(self.modalities)):
            for key, value in reversed(self.modalities[str(modality_index)].items()):
                cast_size = self.modalities[str(modality_index)][key].hidden_units.copy()
                cast_size.insert(0, batch_size)
                outputs[modality_index] = outputs[modality_index].view(tuple(cast_size))
                outputs[modality_index] = value.top_bottom(outputs[modality_index])
        return outputs


    def network_inference(self, inputs):
        """
        Description:
            The method to call to infer the input data to the top layers (or the joint layer if present) and then back to the visible units of the first layers of each modality
        Explanation:
            Calls the function get_input_joint_layer and samples the top layers, to then propagate it back calling the method top_bottom.
        Arguments:
            list of tensors inputs : list of tensors of the input data for each modality
        """
        inputs = self.get_input_joint_layer(inputs)
        if self.joint_layer:
            inputs, _ = self.joint_layer.gibbs_sampling( inputs, 1)
        outputs= self.top_bottom(inputs)
        return outputs
