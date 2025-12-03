from builtins import range
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
        y_hat = X[i] @ W                    # raw scores vector of size C
        y_exp = np.exp(y_hat - y_hat.max()) # Subtract max (for numerical stability), then exponentiate
        softmax = y_exp / y_exp.sum()       # standardize to produce softmax probabilities
        loss -= np.log(softmax[y[i]])       # y[i] holds label for i-th example. We extract the prob. of the correct class
                                             # and accumulate the negative log prob. to loss
        softmax[y[i]] -= 1                  # update for gradient. Remember gradient is P_j - I(j = y_i)
        dW += np.outer(X[i], softmax)       # Multiply gradient with input vector to get contributuion to dW from i-th example

    loss = loss / num_train + reg * np.sum(W**2)    # average loss and regularize 
    dW = dW / num_train + 2 * reg * W               # finish calculating gradient

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X @ W   # Vectorized probabilities of shape (N, C)
    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # Subtract max, vectorized. Note each row is an example, we take off max of each class in all examples.
    softmax = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)  # Vectorized softmax probabilities. 
    loss -= np.sum(np.log(softmax[np.arange(num_train), y]))  # Vectorized loss calculation. Extract correct class probabilities for all examples at once.
    softmax[np.arange(num_train), y] -= 1  # Vectorized gradient update. Subtract 1 from correct class probabilities for all examples at once.
    dW = X.T @ softmax  # Vectorized gradient calculation.
    
    loss = loss / num_train + reg * np.sum(W**2)
    dW = dW / num_train + 2 * reg * W
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW
