import torch
import torch.nn as nn
from competitionZone import CompetitionZone


class TopkLayer1d(nn.Module):
    def __init__(self, input_size, size, neurons_per_zone, stride=1, k=1):
        super().__init__()
        self.k = k
        self.frozen = False
        self.size = size
        self.stride = stride
        num_zones = (input_size - (size-1)) // stride
        # Create the competition zones according to how this layer is defined
        self.zones = [CompetitionZone(neurons_per_zone, self.size, self.k) for _ in range(num_zones)]
        #print("ZONES: ", self.zones)
        self.responses = torch.zeros((num_zones, neurons_per_zone))

    def forward(self, x):
        # Pass data into each competition zone
        x = torch.flatten(x)
        for i in range(len(self.zones)):
            self.responses[i] = self.zones[i](x[self.stride*i:self.stride*i + self.size])
        print("RESPONSES: ", self.responses)
        return self.responses


class TopkLayer2d(nn.Module):
    def __init__(self, height, width, size, neurons_per_zone, stride=1, k=1):
        super().__init__()
        self.k = k
        self.frozen = False
        self.size = size
        self.stride = stride
        self.num_zones = ((height-(size-1)) * (width - (size-1))) // stride
        self.num_width = (width - (size-1)) // stride
        self.num_height = (height - (size-1)) // stride
        # Create the competition zones according to how this layer is defined
        self.zones = [CompetitionZone(neurons_per_zone, self.size**2, self.k) for _ in range(self.num_zones)]
        #print("ZONES: ", self.zones)
        self.responses = torch.zeros((self.num_zones, neurons_per_zone))

    def forward(self, x):
        # Pass data into each competition zone
        for i in range(len(self.zones)):
            start_row = i // self.num_width*self.stride
            start_col = i % self.num_height * self.stride
            self.responses[i] = self.zones[i](x[start_row:start_row + self.size, start_col:start_col + self.size])
        print("RESPONSES: ", self.responses)
        return self.responses