from builtins import range
import numpy as np
from random import shuffle

from numpy.lib.npyio import savez_compressed
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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

    结构SVM损失函数，幼稚实现（带循环）。

    输入有D维，C个类，我们在有N个样例的minibatch小批量上操作。

    输入：
    - W：包含权重的形状为（D,C）的numpy数组
    - X：包含小批量数据的形状为(N,D)的numpy数组
    - y: 包含标签label的形状为（N,）的numpy数组；y[i]=c意思是X[i]标签label为c，其中0<=c<C。
    - reg：（浮点）正则强度

    返回一个多元组：
    - 损失，单个的浮点数
    - 权重W的梯度；一个形状和W相同的数组
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]#C
    num_train = X.shape[0]#N
    loss = 0.0
    for i in range(num_train):#对于X中的每一个样例X[i]：
        scores = X[i].dot(W)#X.shape=（1，D），scores.shape=(1,C)
        correct_class_score = scores[y[i]]
        #这里y[i]是样例i的正确的分类的class的label，忘记可以看下本function的说明里关于输入y的说明
        #correct_class_score取出了该分类中我们刚算出来的score
        for j in range(num_classes):#对于每一个分类class：
            if j == y[i]:
                continue #ignore 正确的class，也就是课堂里提到的j!=y[i]的对应具体实现
            #只有错误分类才需要计算loss，也才需要计算margin，对于每一个j计算margin，如果margin>0加给loss
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            #print(margin)
            #计算错误分类的margin
            if margin > 0:
                loss += margin#如果margin大于0，则被记录入loss中，这是课堂里的算法，忘记了就review下
                #也就是L1 hinge loss 
                #changjiang writen on 20210821
                #可参考：https://blog.csdn.net/i_csdn_water/article/details/114649341
                #https://blog.csdn.net/NODIECANFLY/article/details/82927119
                #并看下李航《统计机器学习，第二版》131页7.2.4合页损失函数
                #按照我的记忆课堂及notes里并未讨论如何计算这里的dW,具体推导就是
                #margin = scores[j] - scores[y[i]] + 1 公式，对它求导：
                #
                dW[:,y[i]]+=-X[i,:]#正确分类label列的梯度是减去当前样例即外层循环的X[i,:]
                dW[:,j]+=X[i,:]#margin > 0时，第j列的梯度计算加上当前样例i的X[i,:]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW  /=  num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    # 计算损失函数的梯度并把它存为dW。                                               #
    # 不要先计算损失，然后计算微分，在损失被计算的同时计算微分可能简单点。                 #
    # 结果你可能需要修改某些上面的代码去计算梯度。                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #pass
    dW  +=  reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    结构SVM损失函数，向量化实现。
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    # 实现向量化版本的结构SVM损失，保存loss的结果。                                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #参考：https://blog.csdn.net/alexxie1996/article/details/79184596
    #pass
    num_classes = W.shape[1]#C
    num_train = X.shape[0]  #N
    scores = np.dot(X,W)    #N*C
    print(scores.shape)
    print(y.shape)
    #np.sum(scores-y[scores])
    correct_scores =  scores[np.arange(num_train),y] #500
    correct_scores = np.reshape(correct_scores,(num_train,1))
    print(correct_scores.shape)
    margins = scores - correct_scores +1
    margins[np.arange(num_train),y] = 0
    margins[margins<=0]=0
    loss = np.sum(margins)/num_train
    loss += reg * np.sum(W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #pass
    margins[margins > 0]= 1.0
    row_sum=np.sum(margins,axis=1)
    margins[np.arange(num_train),y]=-row_sum
    dW += np.dot(X.T,margins) / num_train + reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
