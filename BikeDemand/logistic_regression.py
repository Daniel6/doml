import csv
from sklearn import linear_model

def grab_features(filename):
	selected_data = []
	data = open(filename)
	reader = csv.reader(data, delimiter=",")
	for sample in reader:
		selected_data.append(sample[0][11:13], sample[7], sample[8])

x = grab_features("train.csv")