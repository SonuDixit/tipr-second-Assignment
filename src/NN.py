import numpy as np
from layer import Layer
from matplotlib import pyplot as plt


def CE_error(pred_array, true_array):
    assert pred_array.shape == true_array.shape, str(pred_array.shape) + "not equal " + str(true_array.shape)
    if pred_array[np.where(true_array==1)] == 0:
        # print(pred_array, true_array)
        eps = 0.0001
        ind = np.where(pred_array > 0.999)
        pred_array += eps
        pred_array[ind] -= (eps * 2 * (pred_array.shape[0] - 1))
    return np.sum(-np.log(pred_array[np.where(true_array==1)]))
    # return np.sum(-np.log(pred_array) * true_array)


class NN:
    def __init__(self, input_dim=2, hidden_layer=[3], output_dim=2, activation=["relu"], momentum=None):
        assert len(hidden_layer) == len(activation)
        self.layers = [Layer(input_dim, hidden_layer[0], activation=activation[0], momentum=momentum)]
        for i in range(len(hidden_layer) - 1):
            self.layers.append(Layer(hidden_layer[i], hidden_layer[i + 1], activation[i + 1], momentum=momentum))
        self.layers.append(Layer(hidden_layer[-1], output_dim, activation="softmax", momentum=momentum))

    def __str__(self):
        for l in self.layers:
            print(l)
        return

    def forward(self, input_x):
        for l in self.layers:
            input_x = l.forward_prop(input_x)
        return input_x

    def backward(self, incoming_gradient, lr):
        """
        apply grad at softmax layer, then propogate grad backward
        :return:
        """
        self.layers.reverse()
        for l in self.layers:
            incoming_gradient = l.back_prop(incoming_gradient, lr)
        self.layers.reverse()

    def backward_batch(self, incoming_gradient, lr):
        """
        apply grad at softmax layer, then propogate grad backward
        :return:
        """
        self.layers.reverse()
        for l in self.layers:
            incoming_gradient = l.store_grad(incoming_gradient, lr)
        self.layers.reverse()

    def update_weights(self, lr):
        for l in self.layers:
            l.update_w_b(lr)

    # def fit(self, input_x, y_train, epochs=1, lr=0.01, batch_size=1):
    #     assert input_x.shape[0] == y_train.shape[0], "equal train data not passed"
    #     losses = []
    #     for i in range(epochs):
    #         loss = 0
    #         for j in range(input_x.shape[0]):
    #             pred_vals = self.forward(input_x[j])
    #             # calculate loss between input_x,y_train
    #             loss += CE_error(pred_vals, y_train[j])
    #             # for softmax loss is error_vector -
    #             self.backward(pred_vals - y_train[j], lr)
    #         loss /= input_x.shape[0]
    #         losses.append(loss)
    #         print("epoch = " + str(i + 1) + " loss =" + str(loss))
    #     print("training done")
    #     return losses

    def fit_batch(self, input_x, y_train, epochs=1, lr=0.01, batch_size=1):
        assert input_x.shape[0] == y_train.shape[0], "equal train data not passed"
        assert input_x.shape[0] % batch_size == 0, "batch_size must divide num_train_examples, please change it in main file"
        losses = []
        t = np.hstack((input_x,y_train))
        np.random.shuffle(t)
        input_x, y_train = t[:,0:input_x.shape[1]], t[:,input_x.shape[1]:]
        for i in range(epochs):
            j = 0
            loss = 0
            while j < input_x.shape[0]:
                batch_loss = 0
                for k in range(batch_size):
                    pred_vals = self.forward(input_x[j + k])
                    # if 1.0 in list(pred_vals):
                    #     eps = 0.001
                    #     ind = list(pred_vals).index(1)
                    #     pred_vals += eps
                    #     pred_vals[ind] -= (eps * 2 * (pred_vals.shape[0]-1))

                    loss += CE_error(pred_vals, y_train[j + k])
                    self.backward_batch(pred_vals - y_train[j + k], lr)
                # batch_loss /= batch_size
                self.update_weights(lr)
                j += batch_size
            loss /= input_x.shape[0]
            losses.append(loss)
            print("epoch = " + str(i + 1) + " loss =" + str(loss))

        print("training done")
        return losses

    # def fit_batch2(self, input_x, y_train, epochs=1, lr=0.01, batch_size=1):
    #     assert input_x.shape[0] == y_train.shape[0], "equal train data not passed"
    #     losses = []
    #     for i in range(epochs):
    #         loss = 0
    #         for j in range(input_x.shape[0]):
    #             pred_vals = self.forward(input_x[j])
    #             loss += CE_error(pred_vals, y_train[j])
    #             self.backward_batch(pred_vals - y_train[j], lr)
    #         loss /= input_x.shape[0]
    #         losses.append(loss)
    #         print("epoch = " + str(i + 1) + " loss =" + str(loss))
    #         self.update_weights(lr)
    #     print("training done")
    #     return losses

    def predict(self, input_x):
        pred_vals = []
        for j in range(input_x.shape[0]):
            pred_vals.append(self.forward(input_x[j]))
        return np.argmax(np.asarray(pred_vals),axis=1)


if __name__ == "__main__":
    net = NN()
    input_x = np.asarray([[1, 0], [1, 1], [0, 1], [0, 0]])
    y_train = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]])

    # input_x = np.asarray([[1,0]])
    # y_train = np.asarray([[0,1]])
    l = net.fit_batch(input_x, y_train, epochs=200)
    # print(net.predict(input_x))
    plt.plot(l)
    plt.show()
