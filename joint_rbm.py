#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

'''Implementation of a convolutional joint RBM, to be included in a DBN framework '''

class joint_rbm(nn.Module):
    def __init__(self, name, type, gaussian_units, visible_units, filters_properties, **kwargs):
        '''
        Description:
            The method to call to instanciate the joint RBM object
        Explanation:
            If the visible units are binary, the probability of activation is given by sigmoid(input).
            Should they be gaussian, we consider the probability of activation the input alone.
        Arguments:
            string name : the name of the joint convolutional rbm
            string type : the type of the rbm, either 'convolutional' or 'fully_connected'
            bool gaussian_units : the bool determining the nature of the visible units
            list visible_units : list of list of integers [[d0_modality0,...,dn_modality0],...,[d0_modalityn,...,dn_modalityn]] representing the size of each modality accross each dimension
            filters_properties : list of dictionaries of which keys are the label of the filter property ('f_number', 'f_width', 'f_height', 'stride', 'padding')
                                                               values is the corresponding filter parameter for each filter property
        '''
        super(joint_rbm, self).__init__()
        self.name         = name
        self.gaussian_units = gaussian_units
        self.n_modalities = len(visible_units)
        self.device = 'cpu'

        if type=='convolutional':
            self.filters_parameters = kwargs['filters_properties']
            height = int((visible_units[0][1]-self.filters_parameters[0]['f_height'])/self.filters_parameters[0]['stride'] +1)
            width  = int((visible_units[0][2]-self.filters_parameters[0]['f_width']) /self.filters_parameters[0]['stride'] +1)
            self.hidden_units    = [self.filters_parameters[0]['f_number'], height, width]
            self.parameters   = {
                'h_bias'   : nn.Parameter((torch.ones(hidden_units[0])*-4)),
                'h_bias_m' : nn.Parameter(torch.zeros(hidden_units[0]))
            }
            for index in range(self.n_modalities):
                self.parameters[str(index)] = OrderedDict()
                self.parameters[str(index)]['weights_m'] = nn.Parameter((torch.zeros(self.filters_parameters[index]['f_number'], visible_units[index][0], self.filters_parameters[index]['f_height'], self.filters_parameters[index]['f_width'] )))
                self.parameters[str(index)]['v_bias_m']  = nn.Parameter(torch.zeros(visible_units[index][0]))
                self.parameters[str(index)]['weights'] = nn.Parameter((torch.randn(self.filters_parameters[index]['f_number'], visible_units[index][0], self.filters_parameters[index]['f_height'], self.filters_parameters[index]['f_width'] )*0.01))
                self.parameters[str(index)]['v_bias']  = nn.Parameter(torch.ones(visible_units[index][0])*0.01)

        elif type=='fully_connected':
            self.hidden_units = kwargs['hidden_units']
            self.parameters   = {
                'h_bias'   : nn.Parameter((torch.ones(self.hidden_units)*-4)),
                'h_bias_m' : nn.Parameter(torch.zeros(self.hidden_units))
            }
            for parameter in range(self.n_modalities):
                self.parameters[str(parameter)] = OrderedDict()
                self.parameters[str(index)]['weights_m'] = nn.Parameter((torch.zeros(visible_units[index],self.hidden_units)))
                self.parameters[str(index)]['v_bias_m']  = nn.Parameter(torch.zeros(visible_units[index]))
                self.parameters[str(index)]['weights'] = nn.Parameter((torch.randn(visible_units[index],self.hidden_units))*0.01)
                self.parameters[str(index)]['v_bias']  = nn.Parameter(torch.ones(visible_units[index])*0.01)

    def to_device(self, device):
        '''
        Description:
            The method that is called when moving the network accross devices.
        Explanation:
            The method .to(device) is called for each parameter of the layer.
        Arguments:
            int device : the device to which the network is moved to.
        '''
        self.device = device
        self.parameters['h_bias']   = self.parameters['h_bias'].to(device)
        self.parameters['h_bias_m'] = self.parameters['h_bias_m'].to(device)
        for index in range(self.n_modalities):
            self.parameters[str(index)]['weights'] = self.parameters[str(index)]['weights'].to(device)
            self.parameters[str(index)]['v_bias']  = self.parameters[str(index)]['v_bias'].to(device)
            self.parameters[str(index)]['weights_m'] = self.parameters[str(index)]['weights_m'].to(device)
            self.parameters[str(index)]['v_bias_m']  = self.parameters[str(index)]['v_bias_m'].to(device)

    def initialisation(self, model):
        '''
        Description:
            The method that is called when restoring the parameters from a saved file.
        Explanation:
            The default parameters dictionnary's values are being replaced by the corresponding model parameters.
        Arguments:
            pytorch .statedict() model : the state dictionnary of which values have to be restored.
        '''
        self.parameters['h_bias']   = nn.Parameter(model['h_bias'])
        self.parameters['h_bias_m'] = nn.Parameter(model['h_bias_m'])
        for index in range(self.n_modalities):
            self.parameters[str(index)]['weights'] = nn.Parameter(model[str(index)]['weights'])
            self.parameters[str(index)]['v_bias']  = nn.Parameter(model[str(index)]['v_bias'])
            self.parameters[str(index)]['weights_m'] = nn.Parameter(model[str(index)]['weights_m'])
            self.parameters[str(index)]['v_bias_m']  = nn.Parameter(model[str(index)]['v_bias_m'])

    ################################## proablitiy of activation of the states methods ##################################

    def get_hidden_probability(self, modalities):
        '''
        Description:
            The method to call to compute the propability of activation of hidden units given visible units's states.
        Explanation:
            The hidden units are considered to be binary, hence the probability of activation is given by sigmoid(input)
        Arguments:
            list of tensors modalities : the list of tensors is what the joint layer processes. Hence, we want to use only lists.
        '''
        h_input = self.parameters['h_bias']
        for index, modality in enumerate(modalities):
            if self.type == 'convolutional':
                h_input += F.conv2d(modality, self.parameters[str(index)]['weights'], bias = False)
            else:
                h_input += F.linear(modality, self.parameters[str(index)]['weights'].t(), bias = False)
        p_h  = F.sigmoid(h_input)
        return p_h

    def get_visible_probability(self, h):
        '''
        Description:
            The method to call to compute the probability of activation of the visible units given hidden units's states.
        Explanation:
            If the visible units are binary, the probability of activation is given by sigmoid(input).
            If they are gaussian, we consider the probability of activation the input alone.
        Arguments:
            tensor h : the tensor of the hidden states.
                It is not a list of tensors (contrary to the argument of get_hidden_probability) as we are never, in this implementation, facing the case of multiple hidden layer for one visible one.
        '''
        modalities = []
        for modality in range(self.n_modalities):
            if self.type == 'convolutional':
                Wh = F.conv_transpose2d(h, self.parameters[str(modality)]['weights'], bias = self.parameters[str(modality)]['v_bias'])
            else:
                Wh = F.linear(h, self.parameters[str(modality)]['weights'],self.parameters[str(modality)]["v_bias"])
            if self.gaussian_units[modality]:
                if self.gaussian_units:
                    modalities.append(Wh)
                else:
                    p_v = F.sigmoid(Wh)
                    modalities.append(p_v)
        return modalities

    ################################## Computation of the states methods ##################################

    def get_hidden_states(self, modalities):
        '''
        Description:
            The method to call to compute the states of the hidden units given visible unit's states.
        Explanation:
            The hidden units are considered to be binary, hence the state of the units is sampled from sigmoid(input)
        Arguments:
            list of tensors modalities : the list of tensors is what the joint layer processes. Hence, we want to use only lists.
        '''
        h_input = self.parameters['h_bias']
        for index, modality in enumerate(modalities):
            if self.type == 'convolutional':
                h_input += F.conv2d(modality, self.parameters[str(index)]['weights'], bias = False)
            else:
                h_input += F.linear(modality, self.parameters[str(index)]['weights'].t(), bias = False)
        p_h  = F.sigmoid(h_input)
        sample_h = torch.bernoulli(p_h)
        return sample_h

    def get_visible_states(self, h):
        '''
        Description:
            The method to call to compute the states of the visible units given hidden units's states.
        Explanation:
            If the visible units are binary, the probability of activation is given by sigmoid(input).
                The state of the unit is then sampled from the sigmoid distribution.
            Should they be gaussian, we consider the probability of activation the input alone.
                The state of the unit is then sampled from a normal distribution of which mean is the input and variance 1.
        Arguments:
            tensor h : the tensor of the hidden states.
        '''
        modalities = []
        for modality in range(self.n_modalities):
            if self.type == 'convolutional':
                Wh = F.conv_transpose2d(h, self.parameters[str(modality)]['weights'], bias = self.parameters[str(modality)]['v_bias'])
            else:
                Wh = F.linear(h, self.parameters[str(modality)]['weights'],self.parameters[str(modality)]["v_bias"])
            if self.gaussian_units[modality]:
                if self.gaussian_units:
                    sample_v = torch.normal(Wh, 1)
                else:
                    p_v = F.sigmoid(Wh)
                    sample_v = torch.bernoulli(p_v)
            modalities.append(sample_v)
        return modalities

    ################################## Energy functions ##################################

    def compute_energy(self, input_data, output_data, hidden_states, batch_size):
        '''
        Description:
            The method to call to compute the energy difference of the RBM in the state set by the input data (positive phase) and the state set by the data sampled from the RBM (negative phase).
            This energy does not drive the learning, it is only a training observable.
        Explanation:
            Computes the difference of the hidden, visible and joint energy terms for both the positive and negative phases.
        Arguments:
            list of tensors input_data : list of visible input data (positive phase)
            list of tensors output_data : list of visible output data (negative phase)
            dictionnary hidden_states : dictionary holding the hidden states given by the visible input data at the key 'h0' and the hidden states that set the visible output data at key 'hk'
            int batch_size : the size of the batch that is currently processed
        '''
        if self.type=='convolutional':
            energy = (torch.sum(hidden_states['h0']-hidden_states['hk'],(0,2,3))*self.parameters['h_bias']).sum()
            for index in range(self.n_modalities):
                visible_detection = F.conv2d(input_data[index], self.parameters[str(index)]['weights'])*hidden_states['h0'] - F.conv2d(output_data[index], self.parameters[str(index)]['weights'])*hidden_states['hk']
                if self.gaussian_units[index]:
                    visible = (pow(torch.sum(input_data[index],(0,2,3))-self.parameters[str(index)]['v_bias'],2) - pow(torch.sum(output_data[index],(0,2,3))-self.parameters[str(index)]['v_bias'],2))/2
                else:
                    visible = torch.sum(input_data[index]-output_data[index],(0,2,3))*self.parameters[str(index)]['v_bias']
                energy += visible_detection.sum() + visible.sum()
            return (energy/batch_size).to('cpu')
        else:
            energy = (hidden_states['h0']-hidden_states['hk'])*self.parameters['h_bias'].sum()
            for index in range(self.n_modalities):
                visible_detection = hidden_states['h0']*F.linear(input_data[index], self.parameters[str(index)]["weights"].t()) - hidden_states['hk']*F.linear(output_data[index], self.parameters[str(index)]["weights"].t())
                if self.gaussian_units[index]:
                    visible = (pow(input_data[index]-self.parameters[str(index)]['v_bias'],2) - pow(output_data[index]-self.parameters[str(index)]['v_bias'],2))/2
                else:
                    visible = self.parameters[str(index)]["v_bias"]*input_data[index] - self.parameters[str(index)]["v_bias"]*output_data[index]
                energy += visible_detection.sum() + visible.sum()
            return (energy.sum()/batch_size).to('cpu')'''

    ################################## Update functions ##################################

    def get_weight_gradient(self, hidden_vector, visible_vector, index):
        '''
        Description:
            The method to call to compute the weight gradient, that is called when updating the layer's parameters.
        Explanation:
            Computes <vh> for either the positive or negative states
        Arguments:
            tensor hidden_vector : the tensor of the hidden states of a given phase (positive or negative)
            tensor visible_vector : the tensor of the visible states of a given phase (positive or negative)
        '''
        if self.type=='convolutional':
            return torch.transpose(F.conv2d(torch.transpose(visible_vector,1,0), torch.transpose(hidden_vector,1,0), dilation = self.filters_parameters[index]['stride']),1,0).sum(0)
        else:
            return torch.mm(hidden_vector.t(),visible_vector).t().sum(0)

    def get_bias_gradient(self, vector_0, vector_k):
        '''
        Description:
            The method to call to compute the visible bias gradient, that is called when updating the layer's parameters.
        Explanation:
            Computes the difference <vector_0>-<vector_k> for bias update
        Arguments:
            tensor vector_0 : the tensor of the states of the RBM corresponding to the input data (0 steps of Gibbs sampling ran)
            tensor vector_k : the tensor of the states of the RBM after running k steps of Gibbs sampling.
        '''
        if self.type=='convolutional':
            return torch.add(vector_0, -vector_k).sum([0,2,3])/(self.vector_0[1]*self.vector_0[2])
        else:
            return torch.add(vector_0, -vector_k).sum(0)

    ################################## Sampling method #################################

    def gibbs_sampling(self, input_modalities, k):
        '''
        Description:
            The method to call to get the state of the RBM after k steps of Gibbs sampling ( for the update rule that uses the CD_k algorithm)
        Explanation:
            Iterates trhough a loop of length k to determine the states of visible and hidden units.
        Arguments:
            list of tensor input_modalities : the list of the tensors of the different modalities.
            int k : the length of the chain
        '''
        h0 = self.get_hidden_states(input_modalities)
        hk = h0
        with torch.no_grad():
            for _ in range(k):
                modalities_k = self.get_visible_states(hk)
                hk = self.get_hidden_states(modalities_k)
        hidden_states ={'h0':h0,'hk':hk}
        return modalities_k, hidden_states

    ################################## Update method #################################

    def update_parameters(self, learning_rate, momentum, weight_decay, input_data, output_data, hidden_states, batch_size):
        '''
        Description:
            The method to call when updating the parameters of the given layer.
        Explanation:
        For each parameter of the layer, the gradient is computed. Then the momentum, and finally the update term is added to each corresponding parameter.
            According of the update rule of RBM:
                the gradient for the weights is <v0h0>-<vkhk>
                the gradient for the visible bias is <v0>-<vk>
                the gradient for the hidden  bias is <h0>-<hk>
        Arguments:
            float learning_rate: the learning rate of the training
            float momentum: the momentum of the training
            float weight_decay: the weight decay of the training
            list of tensors input_data : list of visible input data
            list of tensors output_data : list of visible output data
            dictionnary hidden_states : dictionary holding the hidden states given by the visible input data at the key 'h0' and the hidden states that set the visible output data at key 'hk'
            int batch_size : the size of the batch that is currently processed
        '''
        with torch.no_grad():
            d_h = self.get_bias_gradient(hidden_states['h0'], hidden_states['hk'])/batch_size
            self.parameters['h_bias_m']  = torch.add(momentum*self.parameters['h_bias_m'], d_h)
            self.parameters['h_bias']   += learning_rate*torch.add(d_h,  self.parameters['h_bias_m'])
            for index in range(self.n_modalities):
                d_v = self.get_bias_gradient(input_data[index],output_data[index], index)/batch_size
                dw_in  = self.get_weight_gradient(hidden_states['h0'], input_data[index], index)
                dw_out = self.get_weight_gradient(hidden_states['hk'], output_data[index], index)
                d_w  = torch.add(dw_in,-dw_out)/batch_size
                self.parameters[str(index)]['weights_m'] = torch.add(momentum* self.parameters[str(index)]['weights_m'], d_w)
                self.parameters[str(index)]['v_bias_m']  = torch.add(momentum* self.parameters[str(index)]['v_bias_m'], d_v)
                self.parameters[str(index)]['weights']  += learning_rate*(torch.add(d_w, self.parameters[str(index)]['weights_m']))+weight_decay*self.parameters[str(index)]['weights']
                self.parameters[str(index)]['v_bias']   += learning_rate*torch.add(d_v, self.parameters[str(index)]['v_bias_m'])
