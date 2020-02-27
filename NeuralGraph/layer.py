import torch as T
from torch import nn
from torch.nn import functional as F
import numpy as np
from .util import lookup_neighbors


class GraphConv(nn.Module):
    ''' Hidden Convolutional layer in a Neural Graph (as in Duvenaud et. al.,
    2015). This layer takes a graph as an input. The graph is represented as by
    three tensors.
    - The atoms tensor represents the features of the nodes.
    - The bonds tensor represents the features of the edges.
    - The edges tensor represents the connectivity (which atoms are connected to
        which)
    It returns the convolved features tensor, which is very similar to the atoms
    tensor. Instead of each node being represented by a num_atom_features-sized
    vector, each node now is represented by a convolved feature vector of size
    conv_width.
    # Example
        Define the input:
        ```python
            atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features))
            bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features))
            edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')
        ```
        The `NeuralGraphHidden` can be initialised in three ways:
        1. Using an integer `conv_width` and possible kwags (`Dense` layer is used)
            ```python
            atoms1 = NeuralGraphHidden(conv_width, activation='relu', bias=False)([atoms0, bonds, edges])
            ```
        2. Using an initialised `Dense` layer
            ```python
            atoms1 = NeuralGraphHidden(Dense(conv_width, activation='relu', bias=False))([atoms0, bonds, edges])
            ```
        3. Using a function that returns an initialised `Dense` layer
            ```python
            atoms1 = NeuralGraphHidden(lambda: Dense(conv_width, activation='relu', bias=False))([atoms0, bonds, edges])
            ```
        Use `NeuralGraphOutput` to convert atom layer to fingerprint

    # Arguments
        inner_layer_arg: Either:
            1. an int defining the `conv_width`, with optional kwargs for the
                inner Dense layer
            2. An initialised but not build (`Dense`) keras layer (like a wrapper)
            3. A function that returns an initialised keras layer.
        kwargs: For initialisation 1. you can pass `Dense` layer kwargs
    # Input shape
        List of Atom and edge tensors of shape:
        `[(samples, max_atoms, atom_features), (samples, max_atoms, max_degrees,
          bond_features), (samples, max_atoms, max_degrees)]`
        where degrees referes to number of neighbours
    # Output shape
        New atom featuers of shape
        `(samples, max_atoms, conv_width)`
    # References
        - [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292)
    '''
    def __init__(self, input_dim, conv_width, max_degree=6):
        super(GraphConv, self).__init__()
        self.conv_width = conv_width
        self.max_degree = max_degree
        self.inner_3D_layers = nn.ModuleList([nn.Linear(input_dim, self.conv_width) for _ in range(max_degree)])
        # for degree in range(max_degree):

    def forward(self, *input, mask=None):
        atoms, bonds, edges = input

        # Create a matrix that stores for each atom, the degree it is
        atom_degrees = (edges != -1).sum(-1, keepdim=True)

        # For each atom, look up the features of it's neighbour
        neighbor_atom_features = lookup_neighbors(atoms, edges, include_self=True)

        # Sum along degree axis to get summed neighbour features
        summed_atom_features = neighbor_atom_features.sum(-2)

        # Sum the edge features for each atom
        summed_bond_features = bonds.sum(-2)

        # Concatenate the summed atom and bond features
        summed_features = T.cat([summed_atom_features, summed_bond_features], dim=-1)

        # For each degree we convolve with a different weight matrix
        new_features = None
        for degree in range(self.max_degree):
            atom_masks_this_degree = (atom_degrees == degree).float()
            new_unmasked_features = F.relu(self.inner_3D_layers[degree](summed_features))
            # Do explicit masking because TimeDistributed does not support masking
            new_masked_features = new_unmasked_features * atom_masks_this_degree

            new_features = new_masked_features if degree == 0 else new_features + new_masked_features

        # Finally sum the features of all atoms
        return new_features


class GraphConv_p(nn.Module):
    def __init__(self, input_dim, conv_width, max_degree=6):
        super(GraphConv_p, self).__init__()
        self.conv_width = conv_width
        self.max_degree = max_degree
        self.inner_3D_layers = nn.ModuleList([nn.Linear(input_dim, self.conv_width) for _ in range(max_degree)])
        # for degree in range(max_degree):

    def forward(self, *input, mask=None):
        atoms, edges = input

        # Create a matrix that stores for each atom, the degree it is
        atom_degrees = (edges != -1).sum(-1, keepdim=True)

        # For each atom, look up the features of it's neighbour
        neighbor_atom_features = lookup_neighbors(atoms, edges, include_self=True)

        # Sum along degree axis to get summed features
        summed_features = neighbor_atom_features.sum(-2)

        # For each degree we convolve with a different weight matrix
        new_features = None
        for degree in range(self.max_degree):
            atom_masks_this_degree = (atom_degrees == degree).float()
            new_unmasked_features = F.relu(self.inner_3D_layers[degree](summed_features))
            # Do explicit masking because TimeDistributed does not support masking
            new_masked_features = new_unmasked_features * atom_masks_this_degree

            new_features = new_masked_features if degree == 0 else new_features + new_masked_features

        # Finally sum the features of all atoms
        return new_features


class Encoder_1(nn.Module):
    def __init__(self, input_dim, conv_width, max_degree=6):
        super(Encoder_1, self).__init__()
        self.conv_width = conv_width
        self.max_degree = max_degree
        self.inner_3D_layers_1 = nn.ModuleList([nn.Linear(input_dim, self.conv_width) for _ in range(max_degree)])
        self.inner_3D_layers_2 = nn.ModuleList([nn.Linear(input_dim, self.conv_width) for _ in range(max_degree)])

    def forward(self,  atoms,  edges , mask=None):

        # Create a matrix that stores for each atom, the degree it is
        atom_degrees = (edges != -1).sum(-1, keepdim=True)

        # For each atom, look up the features of it's neighbour
        neighbor_atom_features = lookup_neighbors(atoms, edges, include_self=False)

        # Sum along degree axis to get summed features
        summed_features = neighbor_atom_features.sum(-2)

        # For each degree we convolve with a different weight matrix
        new_neighbor_features = None
        new_self_features = None
        for degree in range(self.max_degree):
            atom_masks_this_degree = (atom_degrees == degree).float()
            new_unmasked_neighbor_features = F.relu(self.inner_3D_layers_1[degree](summed_features))
            new_unmasked_self_features = F.relu(self.inner_3D_layers_2[degree](atoms))
            # Do explicit masking because TimeDistributed does not support masking
            new_masked_neighbor_features = new_unmasked_neighbor_features * atom_masks_this_degree
            new_masked_self_features = new_unmasked_self_features * atom_masks_this_degree

            new_neighbor_features = new_masked_neighbor_features if degree == 0 else new_neighbor_features + new_masked_neighbor_features
            new_self_features = new_masked_self_features if degree == 0 else new_self_features + new_masked_self_features

        return summed_features, atoms, new_neighbor_features, new_self_features


class Decoder_1(nn.Module):
    def __init__(self, input_dim, conv_width, max_degree=6):
        super(Decoder_1, self).__init__()
        self.conv_width = conv_width
        self.max_degree = max_degree
        self.inner_3D_layers_1 = nn.ModuleList([nn.Linear(input_dim, self.conv_width) for _ in range(max_degree)])
        self.inner_3D_layers_2 = nn.ModuleList([nn.Linear(input_dim, self.conv_width) for _ in range(max_degree)])
        # for degree in range(max_degree):

    def forward(self, *input, mask=None):
        neighbor_features, self_features, edges = input

        # Create a matrix that stores for each atom, the degree it is
        atom_degrees = (edges != -1).sum(-1, keepdim=True)

        # For each degree we convolve with a different weight matrix
        new_neighbor_features = None
        new_self_features = None
        for degree in range(self.max_degree):
            atom_masks_this_degree = (atom_degrees == degree).float()
            new_unmasked_neighbor_features = F.relu(self.inner_3D_layers_1[degree](neighbor_features))
            new_unmasked_self_features = F.relu(self.inner_3D_layers_2[degree](self_features))
            # Do explicit masking because TimeDistributed does not support masking
            new_masked_neighbor_features = new_unmasked_neighbor_features * atom_masks_this_degree
            new_masked_self_features = new_unmasked_self_features * atom_masks_this_degree

            new_neighbor_features = new_masked_neighbor_features if degree == 0 else new_neighbor_features + new_masked_neighbor_features
            new_self_features = new_masked_self_features if degree == 0 else new_self_features + new_masked_self_features

        return new_neighbor_features, new_self_features


class Encoder_2_c(nn.Module):
    def __init__(self, input_dim=200, output_dim=512):
        super(Encoder_2_c, self).__init__()
        self.fp_len = output_dim
        self.inner_3D_layer = nn.Linear(input_dim, self.fp_len)

    def forward(self, atoms):

        # Compute fingerprint
        fingerprint_out = F.softmax(self.inner_3D_layer(atoms))

        # Sum across all atoms
        final_fp_out = fingerprint_out.sum(dim=-2)
        return final_fp_out


class GraphOutput(nn.Module):
    """ Output Convolutional layer in a Neural Graph (as in Duvenaud et. al.,
        2015). This layer takes a graph as an input. The graph is represented as by
        three tensors.
        - The atoms tensor represents the features of the nodes.
        - The bonds tensor represents the features of the edges.
        - The edges tensor represents the connectivity (which atoms are connected to
            which)
        It returns the fingerprint vector for each sample for the given layer.
        According to the original paper, the fingerprint outputs of each hidden layer
        need to be summed in the end to come up with the final fingerprint.
        # Example
            Define the input:
            ```python
                atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features))
                bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features))
                edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')
            ```
            The `NeuralGraphOutput` can be initialised in three ways:
            1. Using an integer `fp_length` and possible kwags (`Dense` layer is used)
                ```python
                fp_out = NeuralGraphOutput(fp_length, activation='relu', bias=False)([atoms0, bonds, edges])
                ```
            2. Using an initialised `Dense` layer
                ```python
                fp_out = NeuralGraphOutput(Dense(fp_length, activation='relu', bias=False))([atoms0, bonds, edges])
                ```
            3. Using a function that returns an initialised `Dense` layer
                ```python
                fp_out = NeuralGraphOutput(lambda: Dense(fp_length, activation='relu', bias=False))([atoms0, bonds, edges])
                ```
            Predict for regression:
            ```python
            main_prediction = Dense(1, activation='linear', name='main_prediction')(fp_out)
            ```
        # Arguments
            inner_layer_arg: Either:
                1. an int defining the `fp_length`, with optional kwargs for the
                    inner Dense layer
                2. An initialised but not build (`Dense`) keras layer (like a wrapper)
                3. A function that returns an initialised keras layer.
            kwargs: For initialisation 1. you can pass `Dense` layer kwargs
        # Input shape
            List of Atom and edge tensors of shape:
            `[(samples, max_atoms, atom_features), (samples, max_atoms, max_degrees,
              bond_features), (samples, max_atoms, max_degrees)]`
            where degrees referes to number of neighbours
        # Output shape
            Fingerprints matrix
            `(samples, fp_length)`
        # References
            - [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292)
    """

    def __init__(self, input_dim=128, output_dim=128):
        super(GraphOutput, self).__init__()
        self.fp_len = output_dim
        self.inner_3D_layer = nn.Linear(input_dim, self.fp_len)

    def forward(self, atoms, bonds, edges):

        # Create a matrix that stores for each atom, the degree it is, use it
        #   to create a general atom mask (unused atoms are 0 padded)
        # We have to use the edge vector for this, because in theory, a convolution
        #   could lead to a zero vector for an atom that is present in the molecule
        atom_degrees = (edges != -1).sum(-1, keepdim=True)
        general_atom_mask = (atom_degrees != 0).float()

        # Sum the edge features for each atom
        summed_bond_features = bonds.sum(-2)

        # Concatenate the summed atom and bond features
        summed_features = T.cat([atoms, summed_bond_features], dim=-1)

        #Compute fingerprint
        fingerprint_out_unmasked = T.tanh(self.inner_3D_layer(summed_features))

        # Do explicit masking because TimeDistributed does not support masking
        fingerprint_out_masked = fingerprint_out_unmasked * general_atom_mask

        # Sum across all atoms
        final_fp_out = fingerprint_out_masked.sum(dim=-2)
        return final_fp_out


class GraphOutput_p(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super(GraphOutput_p, self).__init__()
        self.fp_len = output_dim
        self.inner_3D_layer = nn.Linear(input_dim, self.fp_len)

    def forward(self, atoms, edges):

        # Create a matrix that stores for each atom, the degree it is, use it
        #   to create a general atom mask (unused atoms are 0 padded)
        # We have to use the edge vector for this, because in theory, a convolution
        #   could lead to a zero vector for an atom that is present in the molecule
        atom_degrees = (edges != -1).sum(-1, keepdim=True)
        general_atom_mask = (atom_degrees != 0).float()

        #Compute fingerprint
        fingerprint_out_unmasked = T.tanh(self.inner_3D_layer(atoms))

        # Do explicit masking because TimeDistributed does not support masking
        fingerprint_out_masked = fingerprint_out_unmasked * general_atom_mask

        # Sum across all atoms
        final_fp_out = fingerprint_out_masked.sum(dim=-2)
        return final_fp_out


class GraphPool(nn.Module):
    """ Pooling layer in a Neural graph, for each atom, takes the max for each
        feature between the atom and it's neighbours
        # Input shape
            List of Atom and edge tensors of shape:
            `[(samples, max_atoms, atom_features), (samples, max_atoms, max_degrees,
              bond_features), (samples, max_atoms, max_degrees)]`
            where degrees referes to number of neighbours
        # Output shape
            New atom features (of same shape:)
            `(samples, max_atoms, atom_features)`
        """
    def __init__(self):
        super(GraphPool, self).__init__()

    def forward(self, atoms, edges):
        # For each atom, look up the featues of it's neighbour
        neighbor_atom_features = lookup_neighbors(atoms, edges, maskvalue=-np.inf, include_self=True)
        # For each atom, look up the featues of it's neighbour
        max_features = neighbor_atom_features.max(dim=2)[0]
        atom_degrees = (edges != -1).sum(dim=-1, keepdim=True)
        general_atom_mask = (atom_degrees != 0).float()
        return max_features * general_atom_mask
