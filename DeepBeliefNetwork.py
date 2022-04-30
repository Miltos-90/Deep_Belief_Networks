import numpy as np
from random import sample
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm.notebook import tqdm
from RestrictedBoltzmannMachine import RestrictedBoltzmannMachine as RBM

class DeepBeliefNetwork(torch.nn.Module):
    ''' Deep Belief Network class'''
    
    def __init__(self, layer_nodes, no_labels, lr, momentum, weight_decay, seed):
        '''
        Initialisation function for the Deep Belief Network
        Inputs:
            layer_nodes:  List of integers indicating the number of neurons in each layer of the DBN
            no_labels:    Number of labels on the dataset [scalar integer]
            lr:           Learning rate for the RBMs [scalar float]
            momentum:     Momentum for the RBMs [scalar float]
            weight_decay: Weight decay for the RBMs [scalar float]
        '''
        
        super().__init__()
        
        self.RBMs         = []     # List to hold RBM objects
        self.no_layers    = 0      # Counter for the number of layers in the DBN
        self.no_labels    = no_labels # Number of labels in the dataset
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        
        # Loop over all layers
        for out_nodes, in_nodes in zip(layer_nodes[1:], layer_nodes[:-1]):

            # Make number of labels (top layer: 10 / else: 0)
            if out_nodes == layer_nodes[-1]:
                labels = no_labels
            else:
                labels = 0
            
            
            # Initialise the RBM
            rbm = RBM(hidden_units  = out_nodes,
                      visible_units = in_nodes + labels,
                      learning_rate = self.lr[0],
                      momentum      = self.momentum[0],
                      weight_decay  = self.weight_decay[0],
                      no_labels     = labels,
                      seed          = seed)

            # Add to list
            self.RBMs.append(rbm)

            # Increment no. layers
            self.no_layers += 1
    
    
    def to(self, device):
        '''
        Uploads DBN (RBMs) to device
        Inputs:
            device:  Device [string]
        '''
        
        # Put RBMs to device
        for rbm in self.RBMs: 
            rbm.to(device)
            
        return
    
    
    def update_rbm_learning_params(self):
        '''
        Updates the learning rate, momentum, and weight decay of the RBMs for 'back-fitting' (up-down) algorithm
        '''
        
        if len(self.lr) > 1:
            for rbm in self.RBMs: 
                rbm.lr = Parameter(torch.tensor(self.lr[1]), requires_grad = False)

        if len(self.momentum) > 1:
            for rbm in self.RBMs: 
                rbm.m  = Parameter(torch.tensor(self.momentum[1]), requires_grad = False)        

        if len(self.weight_decay) > 1:
            for rbm in self.RBMs: 
                rbm.wd = Parameter(torch.tensor(self.weight_decay[1]), requires_grad = False)
        
        return 
    
    
    def layerwise_train(self, train_loader, val_loader, epochs):
        '''
        Performs the greedy layer-wise training of the RBMs comprising the DBN
        Inputs:
            train_loader:  Pytorch dataloader object
            val_loader:    Pytorch dataloader object
            epochs:        Epochs to train for [scalar integer]
        '''
        
        # Make matrices to hold reconstruction error for each batch and each layer
        train_err = np.empty(shape = (epochs, self.no_layers))
        val_err   = np.empty(shape = (epochs, self.no_layers))
        
        for layer in range(self.no_layers): # Loop over all RBMs
        
            # Train / validate layer
            for e in tqdm(range(epochs)):
            
                train_err[e, layer] = self.train_RBM(layer, train_loader, step = True)
                val_err[e, layer]   = self.train_RBM(layer, val_loader,   step = False)
            
        return train_err, val_err
    
    
    def backfit_train(self, train_loader, val_loader, epochs):
        
        # Update training parameters for this phase
        self.update_rbm_learning_params()
        
        # Matrix to hold reconstruction error for training
        train_errors = np.empty(shape = (epochs, self.no_layers))
        val_errors   = np.empty(shape = (epochs, self.no_layers))
        
        # get device
        device = self.RBMs[0].W.device
        
        # Loop over epochs
        for e in tqdm(range(epochs)):

            # Matrices to hold errors for each batch in one epoch
            tr_batch_errors  = np.empty(shape = (len(train_loader), self.no_layers))
            val_batch_errors = np.empty(shape = (len(val_loader), self.no_layers))

            # Train
            for batch_id, (X, y) in enumerate(train_loader):
                
                # Send to device
                X = X.to(device)
                y = y.to(device)
                
                # One-hot encode targets
                y_oh = torch.eye(self.no_labels)[y.long()].squeeze().to(device)
        
                # Fit batch
                tr_batch_errors[batch_id, :] = self.backfit_batch(X, y_oh, step = True)
            
            # Validation
            for batch_id, (X, y) in enumerate(val_loader):
                
                # Send to device
                X = X.to(device)
                y = y.to(device)
                
                # One-hot encode targets
                y_oh = torch.eye(self.no_labels)[y.long()].squeeze().to(device)
        
                # Fit batch
                val_batch_errors[batch_id, :] = self.backfit_batch(X, y_oh, step = False)
                    
            # Compute errors for this epoch
            train_errors[e, :] = np.median(tr_batch_errors, 0)
            val_errors[e, :]   = np.median(val_batch_errors, 0)
            
        return train_errors, val_errors

    
    def backfit_batch(self, X, y, step):
        '''
        Performs the 'back-fitting' algorithm on one batch
        Inputs:
            X:     batch of inputs [tensor (batch_size x input features)]
            y:     batch of targets [tensor (batch_size x 1)]
            step:  Boolean flag indicating whether to update RBM parameters using k-CD
        '''
        
        # Empty lists to hold CD statistics
        p_h_v_pos = [None] * self.no_layers
        h_pos     = [None] * self.no_layers
        p_v_h_neg = [None] * self.no_layers
        v_neg     = [None] * self.no_layers
        v_pos     = [None] * self.no_layers
        p_h_v_neg = [None] * self.no_layers

        # Bottom up pass        
        v_pos[0] = X
        
        for layer, rbm in enumerate(self.RBMs[:-1]):

            p_h_v_pos[layer], h_pos[layer] = rbm.PositivePhase(v_pos[layer])
            v_pos[layer + 1] = h_pos[layer] # Make input for next layer

        # Gibbs sampling for the associative memory
        layer        = self.no_layers - 1
        rbm          = self.RBMs[layer]
        v_pos[layer] = torch.cat((v_pos[layer], y), dim = 1)
        p_h_v_pos[layer], p_h_v_neg[layer], p_v_h_neg[layer], h_pos[layer], v_neg[layer] = rbm.GibbsSampler(v_pos[layer], k = 20)

        # Top down pass
        h_pos_ = v_neg[layer][:, :-self.no_labels]
        for layer, rbm in enumerate(reversed(self.RBMs[:-1])):

            layer = (self.no_layers - 2) - layer # Correct layer index (we are traversing the list in reverse order)
            p_v_h_neg[layer], p_h_v_neg[layer], v_neg[layer] = rbm.NegativePhase(h_pos_)            
            h_pos_ = v_neg[layer]               # Make input for previous layer

        # Update parameters & compute errors for each layers
        error = np.empty(shape = (self.no_layers))
        for layer, rbm in enumerate(self.RBMs):
            if step: 
                rbm.step(p_h_v_pos[layer], p_h_v_neg[layer], v_pos[layer], v_neg[layer])
            error[layer] = rbm.reconstruction_error(v_pos[layer], v_neg[layer])
                    
        return error
    
    
    def predict(self, X):
        '''
        Predicts the label of one sample
        Inputs:
            X:     batch of inputs [tensor (batch_size x input features)]
        '''
            
        # Deterministic up-pass from the image up to the top layer
        for layer, rbm in enumerate(self.RBMs[:-1]):
            _, _, _, X, _ = rbm.GibbsSampler(X, k = 10)

        # Grab top-layer RBM
        rbm = self.RBMs[-1]

        # Make label matrix
        label = torch.eye(self.no_labels).to(rbm.W.device)

        # Make RBM input vector
        v_pos = torch.cat((X.repeat(self.no_labels, 1), label), dim = 1)

        # Gibbs sampling
        p_h_v_pos, p_h_v_neg, p_v_h_neg, h_pos, v_neg = rbm.GibbsSampler(v_pos, k = 10)

        # Compute free energy
        E = rbm.free_energy(v_pos, p_h_v_pos)

        # Grab prediction
        label_pred = int(torch.argmin(E))

        return label_pred

    
    def train_RBM(self, layer, loader, step):
        '''
        Trains an RBM
        Inputs:
            layer:  layer to be trained [scalar integer]
            loader: Dataloader object
            step:   Boolean flag indicating whether or not to perform SGD on the RBM
        '''
        
        # Get RBM of the layer
        rbm = self.RBMs[layer]
            
        # Empty vectors to hold reconstruction errors during training and validation
        err = np.empty(shape = (len(loader)))
        
        # Run k-step Contrastive Divergence
        for idx, (X, y) in enumerate(loader):
            
            # Send to device of the RBM
            X = X.to(rbm.W.device)
            y = y.to(rbm.W.device)
            
            # Transform batch for the RBM of the current layer
            X = self.transform(X, y, layer)
        
            # Run kCD
            err[idx] = float(rbm.ContrastiveDivergence(v = X, k = 1, step = step))
        
        return np.median(err)
    
    
    def transform(self, X, y, layer_idx):
        '''
        Transforms a batch for the input or the RBM at layer <layer_idx>
        Inputs:
            X:         batch of data [tensor (batch_size x input features)]
            layer_idx: identifier of the layer that is being trained [integer]
        '''
        
        # Transform batch
        if layer_idx > 0:
                
            # Loop over all the previous layers
            for layer_pre in range(layer_idx):
                    
                # Get RBM of the current layer
                rbm = self.RBMs[layer_pre]
                    
                # Compute activations of the hidden layer of the RBM
                X = rbm.p(X, rbm.W, rbm.c)
                    
        if layer_idx == self.no_layers - 1: # Final (top) layer

            # One-hot encode the dataset labels
            y_oh = torch.eye(self.no_labels)[y.long()].squeeze().to(y.device)

            # Overwrite inputs of dataset
            X = torch.cat((X, y_oh), dim = 1)
    
        return X
