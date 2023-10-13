import torch
import torch.nn as nn
import random


class DN1(nn.Module):

    def __init__(self, x_size, num_neurons, z_size, topk=1, ztopk=1):
        self.z_response = torch.zeros(z_size)
        self.y_response = torch.zeros(num_neurons)
        self.x_response = torch.zeros(x_size)

        self.z_indx = [num_neurons, num_neurons + z_size+1]
        self.y_indx = [0, num_neurons]
        self.x_indx = None
        
        # Setup neurons for hidden layer
        self.num_neurons_init = 0
        self.max_num_neurons = num_neurons
        self.neurons = torch.zeros((num_neurons + z_size, x_size+num_neurons+z_size))
        self.ages = torch.ones(num_neurons + z_size)

        self.topk = topk
        self.ztopk = ztopk
        self.last_response = None

    def update(self, x, z, supervized_z=None):
        # Combine x and z with y -> inpt_vec
        inpt = torch.hstack([x, self.y_response, z])

        # Dot product inpt_vec (sizex1) with neurons (E x size)
        response = self.neurons @ inpt

        final_rsp = torch.zeros(response.shape)
        
        # Update motor neurons as long as there is a previous hidden layer response
        if self.num_neurons_init != 0:
            # Set motor response to the supervized response
            if supervized_z is not None:
                response[self.z_indx[0]:self.z_indx[1]] = supervized_z
            
            # Update motor neurons
            topk = 1 if self.num_neurons_init <= self.topk else self.ztopk
            values, indxs = torch.topk(response[self.z_indx[0]:self.z_indx[1]], topk+1)
            indxs = self.max_num_neurons + indxs
            final_rsp[indxs[:-1]] = (values[:-1] - values[-1]) / (values[0] - values[-1])
            self.update_neuron_weights(inpt, indxs)

        # Create new neuron if responses not high enough and you have room
        if torch.all(response[self.y_indx[0]:self.y_indx[1]] < 0.5) and self.num_neurons_init < self.max_num_neurons:
            print("CREATING NEW NEURON")
            final_rsp[self.y_indx[0] + self.num_neurons_init] = 1.0
            indxs = [self.y_indx[0] + self.num_neurons_init, self.y_indx[0] + self.num_neurons_init + 1]
            self.num_neurons_init += 1
        # Get topk and rescale hidden layer responses
        else: 
            # Topk for the responses
            topk = 1 if self.num_neurons_init <= self.topk else self.topk
            values, indxs = torch.topk(response[self.y_indx[0]:self.y_indx[1]], topk+1)
            t = 1.0 if torch.any(values[:-1] == values[-1]) else 0.0
            # Rescale responses
            final_rsp[indxs[:-1]] = (values[:-1] - values[-1]) / (values[0] - values[-1] + 0.000000001 * (t * random.random()))

        # Update neurons
        self.update_neuron_weights(inpt, indxs)

        # Set final responses
        self.z_response = final_rsp[self.z_indx[0]:self.z_indx[1]]
        self.y_response = final_rsp[self.y_indx[0]:self.y_indx[1]]
        if self.x_indx is not None:
            self.x_response = final_rsp[self.x_indx[0]:self.x_indx[1]]

        return self.z_response # Return the motor response

    def update_neuron_weights(self, inpt, indxs):
        self.neurons[indxs[:-1]] = ((self.ages[indxs[:-1]] - 1) / self.ages[indxs[:-1]]).reshape((1, -1)) @ self.neurons[indxs[:-1]] + (1/self.ages[indxs[:-1]]).reshape((-1, 1)) @ inpt.reshape((1, -1))
        self.neurons[indxs[:-1]] = self.neurons[indxs[:-1]] / torch.norm(self.neurons[indxs[:-1]], dim=1).reshape((-1, 1))
        self.ages[indxs[:-1]] += 1

    def get_y_weights(self):
        return self.neurons[self.y_indx[0]:self.y_indx[1]]

    def get_z_weights(self):
        return self.neurons[self.z_indx[0]:self.z_indx[1]]
