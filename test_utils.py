import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, roc_auc_score
# from random import shuffle
# import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def hadamard(x, y):
    return x * y


def l1_weight(x, y):
    return np.absolute(x - y)


def l2_weight(x, y):
    return np.square(x - y)


def concate(x, y):
    return np.concatenate((x, y), axis=1)


def average(x, y):
    return (x + y) / 2


def node_classification(labels, node_vector):

    if len(node_vector) == 2:
        B = concate(node_vector[0], node_vector[1])
    else:
        B = node_vector[0]

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf_accu = []
    kf_auc = []
    kf_f1 = []
    labels = np.asarray(labels)
    n = len(labels)
    i = 1
    for train_index, test_index in kf.split(range(n)):
        y_train, y_test = labels[train_index], labels[test_index]
        x_train, x_test = B[train_index], B[test_index]

        clf = LogisticRegression(class_weight='balanced', random_state=1)
        clf.fit(x_train, y_train)
        test_preds = clf.predict(x_test)
        accuracy = clf.score(x_test, y_test)
        f1 = f1_score(y_test, test_preds)

        # print(classification_report(y_test, test_preds))
        kf_accu.append(accuracy)
        kf_auc.append(roc_auc_score(y_test, clf.decision_function(x_test)))
        kf_f1.append(f1)
        i += 1
    # print("==========LR=========")
    print("LR Average Accuracy: %f" % (np.mean(kf_accu)))
    print("LR Average auc: %f" % (np.mean(kf_auc)))
    print("LR Average f1: %f" % (np.mean(kf_f1)))
    return np.mean(kf_accu),np.mean(kf_f1), np.mean(kf_auc)

def link_prediction(train_edges, test_edges, node_vector, op, random_state=1):
    if len(node_vector) == 2:
        B = node_vector[0]
        W = node_vector[1]
    else:
        B = node_vector[0]
        W = node_vector[0]

    X = op(B[train_edges[:, 0]], W[train_edges[:, 1]])
    Y = train_edges[:, 2]
    testX = op(B[test_edges[:, 0]], W[test_edges[:, 1]])
    trueY = test_edges[:, 2]

    clf = LogisticRegression(class_weight='balanced',
                             random_state=random_state)
    clf.fit(X, Y)
    test_preds = clf.predict(testX)
    # f=open('temp.txt','w')
    # print(test_preds.tolist(),file=f)
#     print(classification_report(trueY, test_preds))
    f1 = f1_score(trueY, test_preds)
    roc = roc_auc_score(trueY, clf.decision_function(testX))
#     print(classification_report(trueY, test_preds))
    print("f1 score: %f" % f1)
    print("roc score: %f" % roc)
    return f1, roc

def link_prediction_all(train_edges, test_edges, node_vector):
    f1list = []
    roclist = []
    # print("==============Link Prediction==============")
    # print('hadamard')
    f1, roc = link_prediction(train_edges, test_edges, node_vector, hadamard)
    f1list.append(f1)
    roclist.append(roc)
    # print("============================")
    # print('l1_weight')
    f1, roc = link_prediction(train_edges, test_edges, node_vector, l1_weight)
    f1list.append(f1)
    roclist.append(roc)
    # print("============================")
    # print('l2_weight')
    f1, roc = link_prediction(train_edges, test_edges, node_vector, l2_weight)
    f1list.append(f1)
    roclist.append(roc)
    # print("============================")
    # print('concate')
    f1, roc = link_prediction(train_edges, test_edges, node_vector, concate)
    f1list.append(f1)
    roclist.append(roc)
    # print("============================")
    # print('average')
    f1, roc = link_prediction(train_edges, test_edges, node_vector, average)
    f1list.append(f1)
    roclist.append(roc)

    for f1 in f1list:
        print(f1)
    print('~~~')
    for roc in roclist:
        print(roc)
    return f1list, roclist

def test_sign_pred(model, test_s_list, test_t_list, true_t_list, use_cuda = False):
    model.eval()
    pmi, sign_prob = model(test_s_list, test_t_list)
    if use_cuda:
        sign_prob = sign_prob.data.cpu().numpy()
    else:
        sign_prob = sign_prob.data.numpy()
# np.savetxt('predict_sign_prob.txt', sign_prob)
    ruc = roc_auc_score(true_t_list, sign_prob)

    sign_prob[sign_prob >= 0.5] = 1
    sign_prob[sign_prob < 0.5] = 0
    f1 = f1_score(true_t_list, sign_prob)

    result = 'f1/ruc: {:.6f} {:.6f}'.format(f1, ruc)
    # print(result)
    return result, f1, ruc