import torch
import torch.nn as nn
import torch.nn.functional as F

class CompetitionZone(nn.Module):
    def __init__(self, num_neurons, size, k):
        super().__init__()
        self.k = k
        self.age = torch.ones(num_neurons)
        self.frozen = False
        self.size = size
        self.layer = nn.Linear(size, num_neurons, bias=False)

    def forward(self, x):
        with torch.no_grad():
            x = torch.flatten(x).detach()
            res = self.layer(x)
            vals, inds = torch.topk(res, self.k + 1, sorted=True)
            responses = self.scale_response(vals, inds, res)
            if not self.frozen:
                self.update_weights(x, inds, responses)
        return responses
    
    def scale_response(self, vals, inds, results):
        responses = torch.zeros_like(results)
        responses[inds[:-1]] = (vals[:-1] - vals[-1]) / (vals[0] - vals[-1])
        return responses
    
    def update_weights(self, x, inds, res):
        # TODO: Add forgetting to make it amnesiac avg
        self.layer.weight[inds[:-1]] = (1.0 - 1.0/self.age[inds[:-1]]) * self.layer.weight[inds[:-1]] + (res[inds[:-1]] * (1.0/self.age[inds[:-1]])).reshape((self.k, 1)) @ x.reshape((1, self.size))
        self.age[inds[:-1]] += 1
    
    def set_k(self, k):
        self.k = k

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False