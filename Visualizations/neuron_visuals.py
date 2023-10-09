import matplotlib.pyplot as plt

# TODO: Visualize how similar the weights are between all neurons in a confusion matrix type of graph
def show_similarity(DN):
    y = DN.get_y_weights()
    z = DN.get_z_weights()
    y_similarity = y @ y.T
    z_similarity = z @ z.T
    print("Y SIMILARITY: ", y_similarity)
    print("Z SIMILARITY: ", y_similarity)

# TODO: Visualize the actual weights of each neuron (AKA relative importance of inputs) in confusion matrix type of graph
def show_all_weights(DN):
    y = DN.get_y_weights()
    z = DN.get_z_weights()

# TODO: Visualize current neuron firing pattern in a square graph
def show_firing(DN):
    y_resp = DN.get_y_response()
    z_resp = DN.get_z_response()
    pass

# TODO: Visualize neurons 3d position if 3d tracked
def show_3d_location(DN, show_firing=False):
    pass

# TODO: Visualize the weights of neuron i: (x, y, z)
def show_neuron_weights(DN, i):
    pass

# TODO: Visualize the receptive field of neuron i (all weights that are non-zero)
def show_neuron_rf(DN, i):
    pass

