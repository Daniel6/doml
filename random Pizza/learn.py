import numpy as np
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
        num_comments.append(post["request_number_of_comments_at_retrieval"])
        num_votes.append(post["requester_upvotes_plus_downvotes_at_retrieval"])
        net_votes.append(post["requester_upvotes_minus_downvotes_at_retrieval"])
        Y.append(post["requester_received_pizza"])
    for i in range(len(num_comments)):
        X.append([num_comments[i], num_votes[i], net_votes[i]])
    return X, [int(x) for x in Y]

def main(argv):
	trX, trY = collectData(json.loads(open("Data/train.json").read()))
	print np.reshape(trX, (len(trX),3))

if __name__ == "__main__":
    main(sys.argv)