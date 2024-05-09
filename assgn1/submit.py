import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses

	features = my_map(X_train)
	labels = y_train
	features = StandardScaler().fit_transform(features)
	model = LogisticRegression(C = 100, max_iter = 500, penalty = 'l2', solver = 'lbfgs', tol = 0.01)
	model.fit(features, labels)
	w = model.coef_.flatten()
	b = model.intercept_[0]
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0
	return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	X = np.flip(np.cumprod(np.flip(2 * X - 1, axis = 1), axis = 1), axis = 1)
	X = np.hstack((X, np.ones((X.shape[0], 1))))
	num_columns = X.shape[1]
	feat = np.empty((X.shape[0], num_columns * (num_columns - 1) // 2))
	idx = 0
	for i in range(num_columns):
		feat[:, idx:idx+num_columns-i-1] = X[:, i+1:] * X[:, i][:, np.newaxis]
		idx += num_columns - i - 1
	
	return feat
