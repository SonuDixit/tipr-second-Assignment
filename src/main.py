import pickle
import argparse
from helper_fns import prepare_train_test_cat_dog, prepare_train_test_mnist
from helper_fns_from_1 import data_read_convert_to_np_array, split_train_test, preprocess_text_data, read_label_from_text_file
from sklearn.metrics import f1_score, accuracy_score
from NN import NN
import numpy as np
import os

def print_res(test_label,pred_lab):
    acc = accuracy_score(test_label, pred_lab)
    macro = f1_score(test_label, pred_lab, average="macro")
    micro = f1_score(test_label, pred_lab, average="micro")
    print("Test Accuracy :: " + str(acc * 100))
    print("Test Macro F1-score :: " + str(macro * 100))
    print("Test Micro F1-score :: " + str(micro * 100))

parser = argparse.ArgumentParser()
parser.add_argument('-test_file', '--test-data', help='path of test file', required=False)
parser.add_argument('-train_path', '--train-data', help='path of train directory', required=False)
parser.add_argument('-dataset', '--dataset', help='data file name Dolphins,PubMed, Twitter', required=False)
parser.add_argument('-config', '--configuration',type=str, nargs='+', help='list of ints', required=False)

args = vars(parser.parse_args())
hidden_nodes =[]
if not args["configuration"] is None:
    for i in range(len(args["configuration"])):
        if i==0:
            hidden_nodes.append(int(args["configuration"][0][1:]))
        elif i == (len(args["configuration"])-1):
            hidden_nodes.append(int(args["configuration"][i][:-1]))
        else:
            hidden_nodes.append(int(args["configuration"][i]))
# print(args.keys())
"""
test-data has been converted to test_data
test-label has been converted to test_label
"""
if args["dataset"] == "MNIST" :
    if not args["train_data"] is None:
        tr_data, tr_label = prepare_train_test_mnist(args["train_data"])
        net_4 = NN(hidden_layer=hidden_nodes,
                   activation=["relu"]*len(hidden_nodes),
                   input_dim=784,
                   output_dim=10,
                   momentum=0.1)
        y_tr_one_hot = np.zeros((tr_label.shape[0], 10))
        for i in range(tr_label.shape[0]):
            y_tr_one_hot[i, tr_label[i]] = 1
        loss = net_4.fit_batch(tr_data, y_tr_one_hot,
                               epochs=15,
                               lr=0.0001,
                               batch_size=200)
        test_data, test_label = prepare_train_test_mnist(args["test_data"])
        pred_lab = net_4.predict(test_data)
        print_res(test_label,pred_lab)
    else:
        test_data,test_label = prepare_train_test_mnist(args["test_data"])
        ## load a NN
        with open("net_mnist.pickle" , "rb") as f:
            net_4 = pickle.load(f)
        pred_lab = net_4.predict(test_data)
        print_res(test_label, pred_lab)
elif args["dataset"] in ["cat-dog" ,"Cat_dog","Cat-dog"]:
    if not args["train_data"] is None:
        tr_data, tr_label = prepare_train_test_cat_dog(args["train_data"])
        net_4 = NN(hidden_layer=hidden_nodes,
                   activation=["relu"]*len(hidden_nodes),
                   input_dim=10000,
                   output_dim=2,
                   momentum=0.1)
        y_tr_one_hot = np.zeros((tr_label.shape[0], 2))
        for i in range(tr_label.shape[0]):
            y_tr_one_hot[i, tr_label[i]] = 1
        loss = net_4.fit_batch(tr_data, y_tr_one_hot,
                               epochs=15,
                               lr=0.0001,
                               batch_size=200)
        test_data, test_label = prepare_train_test_cat_dog(args["test_data"])
        pred_lab = net_4.predict(test_data)
        print_res(test_label, pred_lab)
    else:
        test_data,test_label = prepare_train_test_cat_dog(args["test_data"])
        ## load a NN
        with open("net_cat_dog.pickle","rb") as f:
            net_4 = pickle.load(f)
        pred_lab = net_4.predict(test_data)
        print_res(test_label, pred_lab)
elif args["dataset"] == "Twitter" or args["dataset"] == "twitter":
    if not args["train_data"] is None:
        # tr_data, tr_label = prepare_train_test_cat_dog(args["train_data"])
        tr_data = preprocess_text_data(
            data_path=os.path.join(args["train_data"],"twitter.txt"))
        tr_label = read_label_from_text_file(
            os.path.join(args["train_data"], "twitter_label.txt"))
        tr_label += 1
        net_4 = NN(hidden_layer=hidden_nodes,
                   activation=["relu"]*len(hidden_nodes),
                   input_dim=2845,
                   output_dim=3,
                   momentum=0.9)
        y_tr_one_hot = np.zeros((tr_label.shape[0], 3))
        for i in range(tr_label.shape[0]):
            y_tr_one_hot[i, tr_label[i]] = 1
        loss = net_4.fit_batch(tr_data, y_tr_one_hot,
                               epochs=15,
                               lr=0.0001,
                               batch_size=20)
        # test_data, test_label = prepare_train_test_cat_dog(args["test_data"])
        test_data = preprocess_text_data(
            data_path=os.path.join(args["test_data"], "twitter.txt"))
        test_label = read_label_from_text_file(
            os.path.join(args["test_data"], "twitter_label.txt"))
        test_label += 1
        pred_lab = net_4.predict(test_data)
        print_res(test_label, pred_lab)
    else:
        test_data = preprocess_text_data(
            data_path=os.path.join(args["test_data"], "twitter.txt"))
        test_label = read_label_from_text_file(
            os.path.join(args["test_data"], "twitter_label.txt"))
        test_label += 1
        with open("net_twitter.pickle","rb") as f:
            net_4 = pickle.load(f)
        pred_lab = net_4.predict(test_data)
        print_res(test_label, pred_lab)
elif args["dataset"] in ["PubMed", "pubmed","Pubmed"]:
    if not args["train_data"] is None:
        # tr_data, tr_label = prepare_train_test_cat_dog(args["train_data"])
        tr_data = data_read_convert_to_np_array(
            data_path=os.path.join(args["train_data"],"pubmed.csv"))
        tr_label = data_read_convert_to_np_array(
            os.path.join(args["train_data"], "pubmed_label.csv"))
        net_4 = NN(hidden_layer=hidden_nodes,
                   activation=["relu"]*len(hidden_nodes),
                   input_dim=128,
                   output_dim=3,
                   momentum=0.9)
        y_tr_one_hot = np.zeros((tr_label.shape[0], 3))
        for i in range(tr_label.shape[0]):
            y_tr_one_hot[i, tr_label[i]] = 1
        loss = net_4.fit_batch(tr_data, y_tr_one_hot,
                               epochs=15,
                               lr=0.0001,
                               batch_size=20)
        # test_data, test_label = prepare_train_test_cat_dog(args["test_data"])
        test_data = data_read_convert_to_np_array(
            data_path=os.path.join(args["test_data"], "pubmed.csv"))
        test_label = data_read_convert_to_np_array(
            os.path.join(args["test_data"], "pubmed_label.csv"))
        pred_lab = net_4.predict(test_data)
        print_res(test_label, pred_lab)
    else:
        # test_data,test_label = prepare_train_test_cat_dog(args["test_data"])
        test_data = data_read_convert_to_np_array(
            data_path=os.path.join(args["test_data"], "pubmed.csv"))
        test_label = data_read_convert_to_np_array(
            os.path.join(args["test_data"], "pubmed_label.csv"))
        with open("net_pubmed.pickle","rb") as f:
            net_4 = pickle.load(f)
        pred_lab = net_4.predict(test_data)
        print_res(test_label, pred_lab)
elif args["dataset"] in ["dolphins", "Dolphins", "dolphin", "Dolphin"]:
    if not args["train_data"] is None:
        # tr_data, tr_label = prepare_train_test_cat_dog(args["train_data"])
        tr_data = data_read_convert_to_np_array(
            data_path=os.path.join(args["train_data"],"dolphins.csv"))
        tr_label = data_read_convert_to_np_array(
            os.path.join(args["train_data"], "dolphins_label.csv"))
        net_4 = NN(hidden_layer=hidden_nodes,
                   activation=["relu"]*len(hidden_nodes),
                   input_dim=32,
                   output_dim=4,
                   momentum=0.1)
        y_tr_one_hot = np.zeros((tr_label.shape[0], 4))
        for i in range(tr_label.shape[0]):
            y_tr_one_hot[i, tr_label[i]] = 1
        loss = net_4.fit_batch(tr_data, y_tr_one_hot,
                               epochs=15,
                               lr=0.0001,
                               batch_size=1)
        # test_data, test_label = prepare_train_test_cat_dog(args["test_data"])
        test_data = data_read_convert_to_np_array(
            data_path=os.path.join(args["test_data"], "dolphins.csv"))
        test_label = data_read_convert_to_np_array(
            os.path.join(args["test_data"], "dolphins_label.csv"))
        pred_lab = net_4.predict(test_data)
        print_res(test_label, pred_lab)
    else:
        # test_data,test_label = prepare_train_test_cat_dog(args["test_data"])
        test_data = data_read_convert_to_np_array(
            data_path=os.path.join(args["test_data"], "dolphins.csv"))
        test_label = data_read_convert_to_np_array(
            os.path.join(args["test_data"], "dolphins_label.csv"))
        with open("net_dolph.pickle","rb") as f:
            net_4 = pickle.load(f)
        pred_lab = net_4.predict(test_data)
        print_res(test_label, pred_lab)
else:
    print("dataset should be one of these \n ")
    print("MNIST")
    print("Cat_dog")
    print("dolphin")
    print("pubmed")
