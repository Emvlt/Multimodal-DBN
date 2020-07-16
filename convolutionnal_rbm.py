#!/usr/bin/env python
# coding: utf-8

'''Implementation of a convolutional RBM, to be included in a DBN framework '''

import torch
from torch import nn
import torch.nn.functional as F

class conv_rbm(nn.Module):
    def __init__(self, name, gaussian_units, visible_units, f_height, f_width, f_number, c_factor):
        '''
        Description:
            The method to call to create the convolutional RBM object.
        Explanation:
            The object inherits from nn.Module.
        Arguments:
            string name : the name of the rbm
            bool gaussian_units : boolean to determine the type of the visible units
            list visible_units  : list of integers representing the size of the visible layer accross each dimension
            int f_height : filter's height
            int f_width  : filter's width
            int f_number : number of filters
            int c_factor : pooling factor
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

    ################################## GPU and initialisation methods ##################################
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
        self.parameters['weights']   = self.parameters['weights'].to(device)
        self.parameters['v_bias']    = self.parameters['v_bias'].to(device)
        self.parameters['weights_m'] = self.parameters['weights_m'].to(device)
        self.parameters['v_bias_m']  = self.parameters['v_bias_m'].to(device)

    def initialisation(self,model):
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
        self.parameters['weights'] = nn.Parameter(model['weights'])
        self.parameters['v_bias']  = nn.Parameter(model['v_bias'])
        self.parameters['weights_m'] = nn.Parameter(model['weights_m'])
        self.parameters['v_bias_m']  = nn.Parameter(model['v_bias_m'])

    ################################## proablitiy of activation of the states methods ##################################

    def get_hidden_probability(self, v):
        '''
        Description:
            The method to call to compute the propability of activation of hidden units given visible units's states.
        Explanation:
            The hidden units are considered to be binary, hence the probability of activation is given by sigmoid(input)
        Arguments:
            list of tensors v : the list of tensors is what the joint layer processes. Hence, we want to use only lists.
                It is relevant to mention that the list of inputs will have a size different to one only when the joint layer is involved.
        '''
        v = v[0]
        p_h = F.sigmoid(F.conv2d(v, self.parameters['weights'], self.parameters['h_bias'], stride = self.c_factor))
        return  p_h

    def get_visible_probability(self, h):
        '''
        Description:
            The method to call to compute the probability of activation of the visible units given hidden units's states.
        Explanation:
            If the visible units are binary, the probability of activation is given by sigmoid(input).
            Should they be gaussian, we consider the probability of activation the input alone.
        Arguments:
            tensor h : the tensor of the hidden states.
                It is not a list of tensors (contrary to the argument of get_hidden_probability) as we are never, in this implementation, facing the case of multiple hidden layer for one visible one.
        '''
        input_visible = F.conv_transpose2d(h, self.parameters['weights'], bias = self.parameters['v_bias'], stride = self.c_factor)
        if self.gaussian_units:
            p_v = input_visible
        else:
            p_v = torch.sigmoid(input_visible)
        return p_v

    ################################## Compuation of the states methods ##################################

    def get_hidden_states(self, v):
        '''
        Description:
            The method to call to compute the states of the hidden units given visible unit's states.
        Explanation:
            The hidden units are considered to be binary, hence the state of the units is sampled from sigmoid(input)
        Arguments:
            list of tensors v : the list of tensors is what the joint layer processes. Hence, we want to use only lists.
                It is relevant to mention that the list of inputs will have a size different to one only when the joint layer is involved.
        '''

        p_h = self.get_hidden_probability(v)
        h = torch.bernoulli(p_h)
        return h

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
        p_v = self.get_visible_probability(h)
        if self.gaussian_units:
            v = torch.normal(p_v,1)
        else:
            v = torch.bernoulli(p_v)
        return [v]

    ################################## Computation of energy method ##################################

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
        hidden_term = torch.sum(hidden_states['h0']-hidden_states['hk'],(0,2,3))*self.parameters['h_bias']
        joint_term = F.conv2d(input_data[0], self.parameters['weights'], stride = self.c_factor)*hidden_states['h0'] - F.conv2d(output_data[0], self.parameters['weights'], stride = self.c_factor)*hidden_states['hk']
        if self.gaussian_units:
            visible_term = (pow(torch.sum(input_data[0],(0,2,3))-self.parameters['v_bias'],2) - pow(torch.sum(output_data[0],(0,2,3))-self.parameters['v_bias'],2))/2
        else:
            visible_term = torch.sum(input_data[0]-output_data[0],(0,2,3))*self.parameters['v_bias']
        return (-(visible_term.sum() + hidden_term.sum()  + joint_term.sum() )/batch_size).to('cpu')

    ################################## Gradients computation methods ##################################

    def get_weight_gradient(self, hidden_vector, visible_vector):
        '''
        Description:
            The method to call to compute the weight gradient, that is called when updating the layer's parameters.
        Explanation:
            Computes <vh> for either the positive or negative states
        Arguments:
            tensor hidden_vector : the tensor of the hidden states of a given phase (positive or negative)
            tensor visible_vector : the tensor of the visible states of a given phase (positive or negative)
        '''
        return torch.transpose(F.conv2d(torch.transpose(visible_vector,1,0), torch.transpose(hidden_vector,1,0), dilation = self.c_factor),1,0).sum(0)

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
        return torch.add(vector_0, -vector_k).sum([0,2,3])/(self.vector_0[1]*self.vector_0[2])

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
        [h0] = self.get_hidden_states(input_modalities)
        hk = h0
        for _ in range(k):
            modalities_k = self.get_visible_states(hk)
            [hk] = self.get_hidden_states(modalities_k)
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
        d_v = self.get_bias_gradient(input_data[0],output_data[0])/batch_size
        d_h = self.get_bias_gradient(hidden_states['h0'], hidden_states['hk'])/batch_size
        dw_in  = self.get_weight_gradient(hidden_states['h0'], input_data[0])
        dw_out = self.get_weight_gradient(hidden_states['hk'], output_data[0])
        d_w  = torch.add(dw_in,-dw_out)/batch_size
        self.parameters['weights_m'] = torch.add(momentum* self.parameters['weights_m'], d_w)
        self.parameters['v_bias_m']  = torch.add(momentum* self.parameters['v_bias_m'], d_v)
        self.parameters['weights']  += learning_rate*(torch.add(d_w, self.parameters['weights_m']))+weight_decay*self.parameters['weights']
        self.parameters['v_bias']   += learning_rate*torch.add(d_v, self.parameters['v_bias_m'])
        self.parameters['h_bias_m']  = torch.add(momentum*self.parameters['h_bias_m'], d_h)
        self.parameters['h_bias']   += learning_rate*torch.add(d_h,  self.parameters['h_bias_m'])

    ##################################  Inference methods  ##################################

    def bottom_top(self, visible_states):
        '''
        Description:
            The method to call to infer data to the upper layer.
        Explanation:
            We train the layer n+1 by using the hidden posteriors of the layer n.
            Each layer having gaussian visible states, it is necessary after computing the posteriors to renormalize them so that they have 0 mean and 1 variance.
            This is done to comply with the assumption made on the visible states.
        Arguments:
            list of tensors input_data : list of visible input data
        '''
        p_h = self.get_hidden_probability(visible_states)
        p_h = (p_h-p_h.mean())/p_h.std()
        return p_h

    def top_bottom(self, hidden_states):
        '''
        Description:
            The method to call to infer data to the lower layer.
        Explanation:
            To infer the visible data during the top_bottom pass (top being the more compressed state that the data will be in the network), we do as if
        Arguments:
            list of tensors input_data : list of visible input data
        '''
        v = self.get_visible_probability(hidden_states)
        return v[0]
