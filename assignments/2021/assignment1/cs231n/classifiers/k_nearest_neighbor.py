from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        ##
       # print(X.shape)
       # print(self.X_train.shape)
        ##
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                #pass
               # dists[i][j] is the same with dists[i,j]
                dists[i,j]=np.sqrt(np.sum(np.square(X[i]-self.X_train[j])))
                #dists[i][j]=np.linalg.norm(X[i]-self.X_train[j])
            #print(i)    
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #print("finish")       
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            #pass
            #dists[i,:]=np.sqrt(np.sum(np.square(X[i]-self.X_train)))
            dists[i,:]=np.sqrt(np.sum(np.square(X[i,:]-self.X_train),axis=1))
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        计算X中的每个测试点和self.X_train中的每个训练点的距离，不显式使用循环。
        输入/输出：和compute_distances_two_loops相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #pass
        X2=(np.ones((num_train,1))*np.sum(np.square(X),axis=1)).T
        y2=np.ones((num_test,1))*np.sum(np.square(self.X_train),axis=1)
        #print(X2.shape) 500*5000
        #print(y2.shape) 500*5000
        dists=np.sqrt(X2+y2-2*np.dot(X,self.X_train.T))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        给定测试点和训练点间的距离矩阵，对每个测试点预测标签。
        输入：
        - dists：numpy数组，shape为（测试数,训练数），其中dists[i,j]给出第i个测试点和第j个训练点间的距离
        返回：
        - y:numpy数组，shape为（测试数,），包含测试数据的预测的标签，其中y[i]是测试点X[i]的预测标签
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            # 长度为k的列表，其中保存了第i个测试点的k个最近邻的标签
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #使用距离矩阵找到第i个测试点的k个最近邻，使用self.y_train找到这些邻居的label。保存 #
            #这些label到closet_y.                                                    # 
            #提示：查阅numpy.argsort函数                                              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            #pass
            #changjiang writen on 2021/08/17
            distsort=np.argsort(dists[i,:],axis=0)
            #for j in range(k):
            closest_y=self.y_train[distsort[:k]]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            #pass closest_y【i】=1，2，2，4，5，6 按照距离近远排序，并列出label
            #dict={}
            #for j in range(k):
            #    print(closest_y[j] )
            #    if closest_y[j] in dict :
            #        dict[closest_y[j]]=dict[closest_y[j]]+1
            #    else :
            #        dict[closest_y[j]]=1    
            count=0
            label=0
            # for m in dict:
            #      if  dict[m]>most:
            #          index=m
            #          most=dict[m]
            #      elif  dict[m]==most:
            #          if index>m:
            #              index=m  
            # y_pred[i]=index
            for j in closest_y:
                tmp=0
                for kk in closest_y:
                    tmp+=(kk==j)
                if tmp>count:
                    count=tmp
                    label=j
            y_pred[i]=label
            

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
