import numpy as np

def accuracy_score(y_true, y_pred):
	true = len(y_true)
	pred = len(y_pred)
	t=0


	for i in range (len(y_true)):
		if y_true[i] == y_pred[i]:
			t = t+1 

	acc = t /true

	return acc
