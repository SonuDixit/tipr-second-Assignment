import numpy as np
from activation import Activation as A
from initializer import Initialise as I

def apply_activation_fun(data,activation="relu"):
    if activation=="relu":
        return A.relu(data)
    elif activation == "softmax":
        return A.softmax(data)
    elif activation == "tanh":
        return A.tanh(data)
    elif activation == "softplus":
        return A.softplus(data)
    elif activation == "swish":
        return A.swish(data)
    elif activation == "sigmoid":
        return A.sigmoid(data)

def weight_initializer(input_dim, hidden_nodes, w_init="glorot_normal"):
    if w_init == "glorot_normal":
        return I.glorot_normal(shape=(input_dim, hidden_nodes), seed=123)
    elif w_init == "normal":
        return I.normal(shape=(input_dim, hidden_nodes), seed=123)
def bias_initializer(hidden_nodes,seed=123,b_init="zeros"):
    if b_init == "zeros":
        return np.zeros((hidden_nodes,))

class Layer:
    def __init__(self, input_dim, hidden_nodes, activation="relu",
                 w_init = "glorot_normal", b_init = "zeros",momentum = None):
        np.random.seed(123)
        self.weights = weight_initializer(input_dim,hidden_nodes,w_init)
        self.biases = bias_initializer(hidden_nodes,b_init=b_init)
        self.activation = activation
        self.bias_grad = 0.0
        self.weight_grad = 0.0
        self.sum_bias_grad = 0.0
        self.sum_weight_grad = 0.0
        if momentum is None:
            self.momentum = 0.0
        else:
            self.momentum = momentum

    def forward_prop(self, input_data):
        """
        :param input_data: is a row vector
        :return:
        """
        self.input = input_data
        # print("input dta is",self.input)
        temp = np.matmul(self.weights.T,self.input) + self.biases
        self.sum_of_incoming = temp.reshape((temp.shape[0],))
        self.output = apply_activation_fun(temp,self.activation)
        # print("returning ",self.output)
        return self.output

    def update_w_b(self,lr,grad_clip = False):
        self.sum_bias_grad = self.momentum * self.sum_bias_grad + self.bias_grad
        self.sum_weight_grad = self.momentum * self.sum_weight_grad + self.weight_grad

        if grad_clip :
            threshold = 10 * self.weights.shape[0] * self.weights.shape[1]
            if np.linalg.norm(self.sum_weight_grad)>threshold :
                # print("weight grad explosion, clipping it,  max value", threshold)
                self.sum_weight_grad *= threshold/np.linalg.norm(self.sum_weight_grad)

            threshold = 10 * self.biases.shape[0]
            if np.linalg.norm(self.sum_bias_grad)>threshold :
                # print("bias grad explosion, clipping it, max value",threshold)
                self.sum_bias_grad *= threshold/np.linalg.norm(self.sum_bias_grad)

        self.biases -= lr * self.sum_bias_grad
        self.weights -= lr * self.sum_weight_grad
        #resetting grads after a batch
        self.bias_grad = 0
        self.weight_grad = 0

    def back_prop(self,incoming_grad,lr = 0.01):
        """
        :param incoming_grad: should be a 1d vector
        :return:
        """
        if self.activation !="softmax":
            if self.activation == "sigmoid":
                self.local_grad = A.grad_sigmoid(self.sum_of_incoming)
            elif self.activation == "relu":
                self.local_grad = A.grad_relu(self.sum_of_incoming)
            self.local_grad *= incoming_grad #element wise multiplication
        else:
            self.local_grad = incoming_grad

        temp_to_pass_back = [self.local_grad for _ in range(self.weights.shape[0])]
        temp_to_pass_back = np.asarray(temp_to_pass_back)

        bias_grad = self.local_grad
        weight_grad = np.matmul(self.input.reshape((self.input.shape[0],1)),
                                self.local_grad.reshape((1,self.local_grad.shape[0])))

        back_grad = self.weights * temp_to_pass_back
        self.biases =self.biases - lr * bias_grad
        self.weights = self.weights - lr * weight_grad
        ##propogate gradient to previous layer

        return np.sum(back_grad,axis=1)

    def store_grad(self,incoming_grad,lr = 0.01):
        """
        :param incoming_grad: should be a 1d vector
        :return:
        """
        if self.activation !="softmax":
            if self.activation == "sigmoid":
                self.local_grad = A.grad_sigmoid(self.sum_of_incoming)
            elif self.activation == "relu":
                self.local_grad = A.grad_relu(self.sum_of_incoming)
            elif self.activation == "swish":
                self.local_grad = A.grad_swish(self.sum_of_incoming)
            elif self.activation == "tanh":
                self.local_grad = A.grad_tanh(self.sum_of_incoming)
            # print("local grad is:",self.local_grad)
            self.local_grad *= incoming_grad #element wise multiplication
            # print("incoming grad is:",self.local_grad)
        else:
            self.local_grad = incoming_grad
            # print("for softmax incoming grad is:",self.local_grad)
    
        
        temp_to_pass_back = [self.local_grad for _ in range(self.weights.shape[0])]
        temp_to_pass_back = np.asarray(temp_to_pass_back)

        temp = np.matmul(self.input.reshape((self.input.shape[0],1)),
                                self.local_grad.reshape((1,self.local_grad.shape[0])))
        self.bias_grad += self.local_grad
        # print("temp is ",temp)
        # print("weight grad is ",self.weight_grad)
        self.weight_grad += temp
        # print("weight grad is ",self.weight_grad)
        ##propogate gradient to previous layer

        back_grad = self.weights * temp_to_pass_back
        # if np.linalg.norm(back_grad)>1.0:
        #     print("grad explosion , inside store_grad function of layer")
        return np.sum(back_grad,axis=1)

    def __str__(self):
        return "dense layer,hidden nodes =  " + str(self.weights.shape[1]) +" activation ="+str(self.activation)

if __name__ == "__main__":
    l = Layer(2, 3)
    print(l)
    # assert 2==3, "2 is not equal to 3"
    a = np.asarray([1,2])
    b = np.asarray([[1,2],[3,4],[5,6]])
    print(np.sum(b,axis=1))
    c = np.matmul(b,a.reshape((a.shape[0],1)))
    print(c)