import pickle
import argparse
from helper_fns import prepare_train_test_cat_dog, prepare_train_test_mnist
from helper_fns_from_1 import data_read_convert_to_np_array, split_train_test, preprocess_text_data, read_label_from_text_file
from sklearn.metrics import f1_score, accuracy_score
from NN import NN
import numpy as np

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
        elif i == len((args["configuration"])-1):
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
        acc = accuracy_score(test_label, pred_lab)
        macro = f1_score(test_label, pred_lab, average="macro")
        micro = f1_score(test_label, pred_lab, average="micro")
        print("Test Accuracy :: " + str(acc * 100))
        print("Test Macro F1-score :: " + str(macro * 100))
        print("Test Micro F1-score :: " + str(micro * 100))
    else:
        test_data,test_label = prepare_train_test_mnist(args["test_data"])
        ## load a NN
        with open("net_mnist.pickle" , "rb") as f:
            net_4 = pickle.load(f)
        pred_lab = net_4.predict(test_data)
        acc = accuracy_score(test_label, pred_lab)
        macro = f1_score(test_label, pred_lab, average="macro")
        micro = f1_score(test_label, pred_lab, average="micro")
        print("Test Accuracy :: " + str(acc * 100))
        print("Test Macro F1-score :: " + str(macro * 100))
        print("Test Micro F1-score :: " + str(micro * 100))
        pass
elif args["dataset"] in ["cat-dog" ,"Cat_dog","Cat-dog"]:
    if not args["train_data"] is None:
        tr_data, tr_label = prepare_train_test_cat_dog(args["train_data"])
        net_4 = NN(hidden_layer=hidden_nodes,
                   activation=["relu"]*len(hidden_nodes),
                   input_dim=10000,
                   output_dim=2,
                   momentum=0.1)
        y_tr_one_hot = np.zeros((tr_label.shape[0], 10))
        for i in range(tr_label.shape[0]):
            y_tr_one_hot[i, tr_label[i]] = 1
        loss = net_4.fit_batch(tr_data, y_tr_one_hot,
                               epochs=15,
                               lr=0.0001,
                               batch_size=200)
        test_data, test_label = prepare_train_test_cat_dog(args["test_data"])
        pred_lab = net_4.predict(test_data)
        acc = accuracy_score(test_label, pred_lab)
        macro = f1_score(test_label, pred_lab, average="macro")
        micro = f1_score(test_label, pred_lab, average="micro")
        print("Test Accuracy :: " + str(acc * 100))
        print("Test Macro F1-score :: " + str(macro * 100))
        print("Test Micro F1-score :: " + str(micro * 100))
    else:
        test_data,test_label = prepare_train_test_cat_dog(args["test_data"])
        ## load a NN
        with open("net_cat_dog.pickle","rb") as f:
            net_4 = pickle.load(f)
        pred_lab = net_4.predict(test_data)
        acc = accuracy_score(test_label, pred_lab)
        macro = f1_score(test_label, pred_lab, average="macro")
        micro = f1_score(test_label, pred_lab, average="micro")
        print("Test Accuracy :: " + str(acc * 100))
        print("Test Macro F1-score :: " + str(macro * 100))
        print("Test Micro F1-score :: " + str(micro * 100))
# elif args["dataset"] == "Twitter" or args["dataset"] == "twitter":
#     twit_data = preprocess_text_data(args["test_data"])
#     twit_label = read_label_from_text_file(args["test_label"])
#     twit_label += 1   ###in implementation of BAYEs labels should be 0,1,2,3....
#     with open("twitter_clf.pickle" , "rb") as f:
#         clf = pickle.load(f)
#     pred_lab = clf.predict(twit_data)
#     acc = accuracy_score(twit_label, pred_lab)
#     macro = f1_score(twit_label, pred_lab, average="macro")
#     micro = f1_score(twit_label, pred_lab, average="micro")
#     print("Test Accuracy :: " + str(acc * 100))
#     print("Test Macro F1-score :: " + str(macro * 100))
#     print("Test Micro F1-score :: " + str(micro * 100))
#     pass
# elif args["dataset"] in ["PubMed", "pubmed","Pubmed"]:
#     pubmed_data = data_read_convert_to_np_array(args["test_data"])
#     pubmed_label = data_read_convert_to_np_array(args["test_label"])
#     with open("pubmed_clf.pickle" , "rb") as f:
#         clf = pickle.load(f)
#     pred_lab = clf.predict(pubmed_data)
#     acc = accuracy_score(pubmed_label, pred_lab)
#     macro = f1_score(pubmed_label, pred_lab, average="macro")
#     micro = f1_score(pubmed_label, pred_lab, average="micro")
#     print("Test Accuracy :: "+ str(acc*100))
#     print("Test Macro F1-score :: " + str(macro * 100))
#     print("Test Micro F1-score :: " + str(micro * 100))
#     pass
# elif args["dataset"] in ["dolphins", "Dolphins", "dolphin", "Dolphin"]:
#     dolph_data = data_read_convert_to_np_array(args["test_data"])
#     dolph_label = data_read_convert_to_np_array(args["test_label"])
#     with open("dolph_clf.pickle", "rb") as f:
#         clf = pickle.load(f)
#     pred_lab = clf.predict(dolph_data)
#     acc = accuracy_score(dolph_label, pred_lab)
#     macro = f1_score(dolph_label, pred_lab, average="macro")
#     micro = f1_score(dolph_label, pred_lab, average="micro")
#     print("Test Accuracy :: " + str(acc * 100))
#     print("Test Macro F1-score :: " + str(macro * 100))
#     print("Test Micro F1-score :: " + str(micro * 100))
#     pass
else:
    print("dataset should be one of these \n ")
    print("MNIST")
    print("Cat_dog")
