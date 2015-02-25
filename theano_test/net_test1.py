import theano
from theano import tensor as T
import numpy as np
from load import mnist
from foxhound.utils.vis import grayscale_grid_vis, unit_scale
from scipy.misc import imsave
import json
import sys

def collectData(data):
    #collect the data
    num_comments = []
    num_votes = []
    net_votes = []
    X = []
    Y = []
    for post in data:
        #create list of all the dictionaries for the karma of the post
        X.append([post["requester_number_of_comments_at_request"],
            post["requester_upvotes_plus_downvotes_at_request"],
            post["requester_upvotes_minus_downvotes_at_request"],
            int(post["post_was_edited"]),
            int(post["requester_account_age_in_days_at_request"]),
            post["requester_number_of_posts_at_request"]])
        Y.append(post["requester_received_pizza"])
    return np.reshape(X, (len(X), 6)), np.reshape([[int(x), 0] for x in Y], (len(X), 2))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def model(X, w_h, w_o):
    h = T.nnet.sigmoid(T.dot(X, w_h))
    pyx = T.nnet.softmax(T.dot(h, w_o))
    return pyx

# trX, teX, trY, teY = mnist(onehot=True)
#print(trX)
trX, trY = collectData(json.loads(open("Data/train.json").read()))
 #= collectData(json.loads(open("Data/test.json").read()))

X = T.fmatrix()
Y = T.fmatrix()

# w_h = init_weights((784, 625))
# w_o = init_weights((625, 10))

w_h = init_weights((6, 4))
w_o = init_weights((4, 2))

py_x = model(X, w_h, w_o)
y_x = T.argmax(py_x, axis=1)
# y_x = T.argmax(py_x, axis=1)
# cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w_h, w_o]
updates = sgd(cost, params)
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
sampindex1 = 30
sampindex2 = sampindex1 + 2
# print([trX[sampindex1]], [trY[sampindex2]])
# print(train([trX[sampindex1]], [trY[sampindex2]]))
# print(predict([trX[sampindex1]])[0], trY[sampindex2])
for i in range(100):
    cost = train(trX, trY)
    print(cost)
    # for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
    # for start, end in zip(range(0, trX.shape[0], 202), range(202, trX.shape[0], 202)):
        # print(trY[start:end])
        # cost = train(trX[start:end], trY[start:end])
        # print(cost)
    num_1 = 0
    num_0 = 0
    for x in predict(trX):
        if x==1:
            num_1 += 1
        elif x==0:
            num_0 += 1
        else:
            print("Predicted non-zero", x)
    print("Zeros: " + str(num_0) + "\nOnes: " + str(num_1))

    # print("Accuracy:", np.mean(predict(trX)==trY))
    # print("Accuracy:", np.mean(np.argmax(trY, axis=1) == predict(trX)))
        
        

predictions = predict(trX)
# print(predictions)
# print(trY)
avg = 0
for i in range(len(trX)):
    # print(predictions[i], trY[i], predictions[i]==trY[i][0])
    avg += float(predictions[i]==trY[i][0])/len(trX)
print("Accuracy: " + str(avg))
# print(np.mean(trY == predict(trX)))