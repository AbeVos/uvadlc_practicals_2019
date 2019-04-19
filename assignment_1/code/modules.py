"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    std = 0.0001

    self.params = {
        'weight': std * np.random.randn(in_features, out_features),
        'bias': np.zeros(out_features)
    }
    self.grads = {
        'weight': np.zeros_like(self.params['weight']),
        'bias': np.zeros_like(self.params['bias'])
    }
    #######################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object.
    They can be used in backward pass computation.
    """
    
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    out = self.x @ self.params['weight'] + self.params['bias'][None, :]
    #######################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    self.grads['weight'] = self.x.T @ dout
    self.grads['bias'] = dout.sum(0)  
    dx = dout @ self.params['weight'].T
    #######################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    out = np.max([np.zeros_like(x), x], axis=0)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    dx = np.argmax([np.zeros_like(self.x), self.x], axis=0) * dout
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    exps = np.exp(x - np.max(x, axis=1)[:, None])
    out = exps / exps.sum(1)[:, None]
    self.out = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = []

    for out, d in zip(self.out, dout):
        out = out[:,None]
        out = np.repeat(out, len(out), 1)
        eye = np.eye(len(out))

        dx_ = out * (eye - out.T)

        dx.append(dx_ @ d)

    dx = np.array(dx)

    '''
    eye = np.eye(len(self.out.T))
    out = np.repeat(self.out[..., None], len(self.out.T), -1)

    dx = out * (eye - out.transpose(0, 2, 1))
    print(dx.shape, dout.shape)

    # dx = np.tensordot(dx, dout[..., None], 1)
    dx = np.einsum('ijk,ik->ik', dx, dout)
    '''
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = - np.sum(y * np.log(x + 1e-7))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = - y / (x + 1e-7)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
