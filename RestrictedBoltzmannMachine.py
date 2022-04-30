import numpy as np
from random import sample
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class RestrictedBoltzmannMachine(torch.nn.Module):
    ''' Restricted Boltmann Machine class '''
    
    def __init__(self, hidden_units, visible_units, learning_rate, momentum = 0, weight_decay = 0, seed = 123, no_labels = 0):
        ''' Initialisation function
            Inputs [all scalars]:
                hidden units:   No. hidden units in the RBM 
                visible_units:  No. visible units in the RBM 
                learning_rate : Learning rate for SGD
                momentum:       Momentum for SGD
                weight_decay:   Weight decay for SGD
                seed:           Seed for weight initialisation
        '''
        
        super().__init__()
        
        # Convert to tensors
        learning_rate      = torch.tensor(learning_rate)
        momentum           = torch.tensor(momentum)     
        weight_decay       = torch.tensor(weight_decay)
        
        # Make parameters (to easily move to any device)
        self.lr       = Parameter(learning_rate, requires_grad = False)      
        self.m        = Parameter(momentum,      requires_grad = False)           
        self.wd       = Parameter(weight_decay,  requires_grad = False)
        self.labels   = no_labels
        
        self.M = visible_units
        self.N = hidden_units
        torch.manual_seed(seed) # Reproducibility
                
        # Initialise Weight matrix
        self.W = Parameter(torch.empty(hidden_units, visible_units), requires_grad = False)
        torch.nn.init.normal_(self.W, mean = 0, std = 0.1)
        
        # Initialise bias vector for hidden units
        self.c = Parameter(torch.zeros(1, hidden_units), requires_grad = False)
        
        # Initialise bias vector for visible units
        self.b = Parameter(torch.zeros(1, visible_units), requires_grad = False)
        
        # Internal variables for SGD (w/ momentum)
        self.DW_pre = 0
        self.Db_pre = 0
        self.Dc_pre = 0
    
    
    def ContrastiveDivergence(self, v, k, step = True):
        ''' k-step Contrastive Divergence training algorithm
            Inputs:
                v: batch [tensor (batch size x visible units)]
                k: No. steps for Gibbs sampling [scalar]
                step: Boolean flag indicating whether or not to perform SGD
            Outputs:
                error: Reconstuction error [scalar]
        '''
        
        # Run Gibbs sampling
        p_h_v_pos, p_h_v_neg, _, _, v_neg = self.GibbsSampler(v, k)      

        # Update parameters
        if step:
            self.step(p_h_v_pos, p_h_v_neg, v, v_neg)

        # Compute reconstruction error 
        error = self.reconstruction_error(v, v_neg)
        
        return error

    
    def PositivePhase(self, v_pos):
        ''' Positive phase
            Inputs:
                v_pos: batch [tensor (batch size x visible units)]
            Outputs:
                p_h_v_pos: Probability of hidden unit being active (Positive phase) [tensor (batch size x hidden units)]
                h_pos:     Hidden states (Positive phase) [tensor (batch size x hidden units)]
        '''
        
        p_h_v_pos = RestrictedBoltzmannMachine.p(v_pos, self.W, self.c)
        h_pos     = RestrictedBoltzmannMachine.sample(p_h_v_pos)
        
        return p_h_v_pos, h_pos
    
    
    def NegativePhase(self, h_pos):
        ''' Positive phase
            Inputs:
                h_pos: Hidden states (Positive phase) [tensor (batch size x hidden units)]
            Outputs:
                p_h_v_neg: Probability of hidden unit being active (Negative phase) [tensor (batch size x hidden units)]
                p_v_h_neg: Probability of visible unit being active (Positive phase) [tensor (batch size x visible units)]
                v_neg:     Visible states (Negative phase [tensor (btch size x visible units)]
        '''
        
        # Treat labels (top layer only)
        if self.labels > 0:
                
            # linear layer
            p_v_h_neg = F.linear(h_pos, self.W.t(), self.b)
                
            # Apply softmax to label units
            p_v_h_neg[:, -self.labels:] = F.softmax(p_v_h_neg[:, -self.labels:], dim = 1)
                
            # Apply sigmoid to all other units
            p_v_h_neg[:, :-self.labels] = torch.sigmoid(p_v_h_neg[:, :-self.labels])
                
        else:
            p_v_h_neg = RestrictedBoltzmannMachine.p(h_pos, self.W.t(), self.b) # linear + sigmoid to all
            
        v_neg     = RestrictedBoltzmannMachine.sample(p_v_h_neg)
        p_h_v_neg = RestrictedBoltzmannMachine.p(v_neg, self.W, self.c)
        
        return p_v_h_neg, p_h_v_neg, v_neg
    
    
    def GibbsSampler(self, v_pos, k):
        ''' k-step Gibbs sampler
            Inputs:
                v_pos: batch [tensor (batch size x visible units)]
                k:     No. steps for Gibbs sampling [scalar]
            Outputs:
                p_h_v_pos: Probability of hidden unit being active (Positive phase) [tensor (batch size x hidden units)]
                p_h_v_neg: Probability of hidden unit being active (Negative phase) [tensor (batch size x hidden units)]
                p_v_h_neg: Probability of visible unit being active (Positive phase) [tensor (batch size x visible units)]
                h_pos:     Hidden states (Positive phase) [tensor (batch size x hidden units)]
                v_neg:     Visible states (Negative phase [tensor (btch size x visible units)]
        '''
        
        for _ in range(k):

            # Positive phase
            p_h_v_pos, h_pos = self.PositivePhase(v_pos)
            
            # Negative phase
            p_v_h_neg, p_h_v_neg, v_neg = self.NegativePhase(h_pos)
        
        return p_h_v_pos, p_h_v_neg, p_v_h_neg, h_pos, v_neg


    def step(self, p_h_v_pos, p_h_v_neg, v_pos, v_neg):
        ''' Parameter updates using SGD w/ momentum 
            Inputs:
                p_h_v_pos: Probability of hidden unit being active (Positive phase) [tensor (batch size x hidden units)]
                p_h_v_neg: Probability of hidden unit being active (Negative phase) [tensor (batch size x hidden units)]
                p_v_h_neg: Probability of visible unit being active (Positive phase) [tensor (batch size x visible units)]
                v_neg:     Visible states (Negative phase [tensor (btch size x visible units)])
                v_pos:     Visible states (Negative phase [tensor (btch size x visible units)])
            Outputs: None
        '''
        
        # Compute gradients
        dW = RestrictedBoltzmannMachine.p_HvV(p_h_v_pos, v_pos) - RestrictedBoltzmannMachine.p_HvV(p_h_v_neg, v_neg)
        db = (v_pos - v_neg).mean(dim = 0)
        dc = (p_h_v_pos - p_h_v_neg).mean(dim = 0)

        # Compute parameter updates
        Delta_w = self.m * self.DW_pre + self.lr * dW - self.wd * self.W
        Delta_b = self.m * self.Db_pre + self.lr * db
        Delta_c = self.m * self.Dc_pre + self.lr * dc

        # Step
        self.W += Delta_w
        self.b += Delta_b
        self.c += Delta_c
        
        # Update for next iteration
        self.DW_pre = Delta_w
        self.Db_pre = Delta_b
        self.Dc_pre = Delta_c
    
        return

    def free_energy(self, v, h, w = None):
        ''' Computation of free energy 
            E = - sum_{i in visible} b_i * v_i 
                - sum_{j in hidden} c_j * h_i 
                - sum_{i in visible, j in hidden} v_i, h_j, w_ij
                
            Inputs:
                v: Visible states (Negative phase) [tensor (batch size x visible units)]
                h: Hidden states (Negative phase) [tensor (batch size x visible units)]
                w: RBM weights [tensor (hidden units x visible units)]
            Outputs:
                E: Free energy [tensor (batch size x 1)]
        '''

        if w is None: w = self.W
        
        E = - (torch.matmul(v, self.b.t()) + torch.matmul(h, self.c.t())).squeeze() - (v * torch.matmul(h, w)).sum(dim = 1)

        return E
    
    @staticmethod
    def p(X, W, bias):
        ''' Computation of p(h|v) or p(v|h) 
            Inputs:
                X: Batch [tensor (batch size x visible units) or (batch size x hidden units)]
                W: RBM Weights [tensor (hidden units x visible units)]
            Outputs: 
                p(h|v) or p(v|h) [tensor (batch size x visible units) or (batch size x hidden units)]
        '''
        return torch.sigmoid(F.linear(X, W, bias))

    
    @staticmethod
    def p_HvV(p, v):
        ''' Computation of p(H|v) x v 
            Inputs:
                v: Visible states (Negative phase) [tensor (batch size x visible units)]
                p: Probability of hidden unit being active (Negative phase) [tensor (batch size x hidden units)]
            Outputs: p(H|v) x v [tensor (visible units x visible units)]
        
        '''
        return torch.matmul(p.t(), v)
    
    
    @staticmethod
    def sample(p): 
        ''' Bernoulli sampling given probability p 
            Inputs:
                p: Probability [tensor (hidden units x 1) or (visible units x 1)]
            Outputs:
                Bernoulli sample [tensor (same as p)]
        '''
        return torch.bernoulli(p)
    
    
    @staticmethod
    def reconstruction_error(v_pos, v_neg):
        ''' Computation of reconstruction error 
            Inputs:
                v_pos: Visible units (Positive phase) [tensor (batch size x visible units)]
                v_neg: Visible units (Negative phase) [tensor (batch size x visible units)]
            Outputs:
                error: Average Reconstruction error [scalar]
        '''
        return torch.sum( (v_neg - v_pos) ** 2, dim = 1).mean().cpu().numpy()