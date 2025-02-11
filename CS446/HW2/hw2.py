import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    #pass
    n = x_train.shape[0]
    K = np.array([kernel(xi,xj) for xi in x_train for xj in x_train]).reshape(n,n) 
    K = torch.from_numpy(K)
    a = torch.zeros(n,requires_grad = True)
    
    def loss_func(a):
        return 0.5*torch.sum(torch.outer(a,a) * torch.outer(y_train,y_train)*K) - torch.sum(a)
    
    
    for i in range(num_iters):
        
        loss_func(a).backward()    
        with torch.no_grad():
            a -= lr * a.grad
            #a = torch.clamp(a,min=0)
            a.clamp_(min=0)
            a.grad.zero_()
           
    return a

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    m,n = x_test.shape[0],x_train.shape[0]
    Phi = np.array([kernel(X_test,X_train) for X_test in x_test for X_train in x_train]).reshape(m,n)
    Phi = torch.from_numpy(Phi)
    out = Phi @ (alpha * y_train)
    return out
    
                       
                       

class CAFENet(nn.Module):
    def __init__(self):
        '''
            Initialize the CAFENet by calling the superclass' constructor
            and initializing a linear layer to use in forward().

            Arguments:
                self: This object.
        '''
        super(CAFENet, self).__init__()
        self.fc = nn.Linear(380*240,6)

    def forward(self, x):
        '''
            Computes the network's forward pass on the input tensor.
            Does not apply a softmax or other activation functions.

            Arguments:
                self: This object.
                x: The tensor to compute the forward pass on.
        '''
        x = self.fc(x)
        return x

def fit(net, X, y, n_epochs=5000):
    '''
    Trains the neural network with CrossEntropyLoss and an Adam optimizer on
    the training set X with training labels Y for n_epochs epochs.

    Arguments:
        net: The neural network to train
        X: n x d tensor
        y: n x 1 tensor
        n_epochs: The number of epochs to train with batch gradient descent.

    Returns:
        List of losses at every epoch, including before training
        (for use in plot_cafe_loss).
    '''
    pass

def plot_cafe_loss():
    '''
    Trains a CAFENet on the CAFE dataset and plots the zero'th through 200'th
    epoch's losses after training. Saves the trained network for use in
    visualize_weights.
    '''
    pass

def visualize_weights():
    '''
    Loads the CAFENet trained in plot_cafe_loss, maps the weights to the grayscale
    range, reshapes the weights into the original CAFE image dimensions, and
    plots the weights, displaying the six weight tensors corresponding to the six
    labels.
    '''
    pass

class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        1) a 2D convolutional layer with 1 input channel and 8 outputs, with a kernel size of 3, followed by
        2) a 2D maximimum pooling layer, with kernel size 2
        3) a 2D convolutional layer with 8 input channels and 4 output channels, with a kernel size of 3
        4) a fully connected (Linear) layer with 4 inputs and 10 outputs
        '''
        super(DigitsConvNet,self).__init__()
        self.conv1 = nn.Conv2d(1,8,3)
        self.conv2 = nn.Conv2d(8,4,3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(4,10)
        

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        xb = xb.view(xb.size()[0],1,xb.size()[1],-1) 
        xb = F.relu(self.conv1(xb))
        print("Fist layer",xb.size())
        xb = F.max_pool2d(xb,2)
        print("maxpool",xb.size())
        xb = F.relu(self.conv2(xb))
        xb = xb.view(xb.size()[0],-1)
        xb = self.fc(xb)
        
        return xb

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = []
    test_losses = []
    
    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss
    with torch.no_grad():
        #train_losses.append(hw2_utils.epoch_loss(net, loss_func, train_dl))
        #test_losses.append(hw2_utils.epoch_loss(net, loss_func, test_dl))
        train_losses += [hw2_utils.epoch_loss(net, loss_func, train_dl)]
        test_losses += [hw2_utils.epoch_loss(net, loss_func, test_dl)]
    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss

    for i in range(n_epochs):
        for x, y in train_dl:
            hw2_utils.train_batch(net, loss_func, x, y, optimizer)
            
        with torch.no_grad():
            #train_losses.append(hw2_utils.epoch_loss(net, loss_func, train_dl))
            #test_losses.append(hw2_utils.epoch_loss(net, loss_func, test_dl))
            train_losses += [hw2_utils.epoch_loss(net, loss_func, train_dl)]
            test_losses += [hw2_utils.epoch_loss(net, loss_func, test_dl)]
            
    return train_losses, test_losses
