
import numpy as np

def rmse(y_true, y_pred):
	# Write your code   float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

	rmse_ = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

	return round(rmse_,3)
