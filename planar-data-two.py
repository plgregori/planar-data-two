import numpy as np
import matplotlib.pyplot as plt

# We start by generating some data, in the form of 300 red, yellow, and blue dots, arranged in a fuzzy spiral shape

N = 100 # number of points per class (i.e. the 100 dots for each color)
D = 2 # dimensionality of each example (i.e. the two coordinates)
K = 3 # number of classes (i.e. the three colors)
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels (stored as unsigned 8-bit integers)
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # evenly spaced radial coordinate of the dots
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # evenly spaced angular coordinate of the dots, plus random noise
  
  #X[ix] = np.c_[r, np.sin(20*r)+2*j+np.random.randn(N)*0.2]
  X[ix] = np.c_[r*np.cos(t), r*np.sin(t)]
  y[ix] = j
# lets visualize the data:
h = 0.02 # this is only necessary for later

x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
plt.scatter(X[:, 0], X[:, 1],c=y, s=40, cmap=plt.cm.Spectral,edgecolors='black')
plt.title("Classification dataset", fontdict = {'fontsize' : 15})
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

# np.linspace(a,b,N) returns N evenly spaced number from a to b
# np.c_[array1,array2] stacks array1 on top of array2 and then takes the transpose
# X[:, 0] means all the first column elements of X
# c=y assigns three different colors to the dots of the scatter plot
# s=40 sets the size of the dots
# cmap specifies the color map

# Normally we would want to preprocess the dataset so that each feature has zero mean and unit standard deviation, but in this case the features are already in a nice range from -1 to 1, so we skip this step.

#We now train the linear classifier

# We start by initializing the weights to random values and the bias to zero

W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# We introduce two hyperparameters. step_size is the usual step parameter for gradient descent. reg is the constant appearing in the regularization of the loss function

step_size = 1e-0
reg = 1e-3 # regularization strength

# We now implement the gradient descent loop

num_examples = X.shape[0]
for i in range(200):

  # First we evaluate the scores
  scores = np.dot(X, W) + b

  # Then the normalized exponential probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

  # Then the loss function in terms of logarithmic probabilities + regularization
  correct_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(correct_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  # And we print the loss every 10 iterations
  if i % 10 == 0:
        print("iteration %d: loss %f" % (i, loss))

  # We compute the derivatives of the loss w.r.t. the scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples

  # And w.r.t. the parameters W and b
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)

  dW += reg*W # This is to include also the regularization contribution

  # We update the parameters
  W += -step_size * dW
  b += -step_size * db

# We evaluate the accuracy of our linear classifier.
# First using argmax, we select for each row the position of the highest value. e.g.: np.argmax([[0.3, 2.2, 4.5]]) = 2
# Then we use predicted_class == y to create an array of ones and zeros depending on whether the elements of predicted_class and y match. Note that this works only if we defined them as numpy arrays
# Finally we use np.mean to sum over all the 'one' entries and divide by the total number of entries

scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))

# Finally, we print the decision boundaries associated with our prediction
# First, we introduce minimum and maximum values for the two coordinates    
# With np.arange we create two arrays of x and y coordinates, evenly spaced by h and going from the minimum to the maximum values
# With np.meshgrid we combine the two arrays into two matrices of coordinates xx and yy, which define a grid
# With .ravel() flattens xx and yy into arrays, and with c_ we stack them and transpose them to create something with the shape of a dataset
# We then multiply this with W and sum b to obtain a prediction on the new dataset
# With np.argmax we select the strongest class, and reshape it in such a way that it looks like a coordinate matrix like xx or yy
# plt.contourf takes as arguments two coordinate matrices, which generate a grid, and a color matrix, which assigns to each site in the grid a color
# By setting alpha=0.8 we make the prediction slightly less transparent than the training dots

Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='black')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Linear classifier decision boundaries", fontdict = {'fontsize' : 15})
plt.show()

# We now train the 2-layer neural network
# We start by setting some hyperparameters and initializing the parameters

h = 100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
step_size = 1
reg = 1e-3 # regularization strength

# We now implement the gradient descent

num_examples = X.shape[0]
for i in range(10000):
  
  # The scores are now computed by first evaluating the hidden layer with ReLU activation
  hidden_layer = np.maximum(0, np.dot(X, W) + b)
  scores = np.dot(hidden_layer, W2) + b2
  
  # Probabilities and loss function are computed exactly like before
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2) # Note the extra regularization coming from the hidden layer
  loss = data_loss + reg_loss
  if i % 1000 == 0:
    print("iteration %d: loss %f" % (i, loss))
  
  # Also the first part of backpropagation is the same as before
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples
  
  # And the backpropagation of the output layer is as before, replacing X with hidden_layer
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # The new derivatives are obtained in a straightforward way (see notes)
  dhidden = np.dot(dscores, W2.T)
  # In order to implement the derivative of the ReLU function, we first create the boolean matrix 'hidden_layer <= 0'
  # Then we use it to set to zero all entries of dhidden corresponding to negative values of hidden_layer
  dhidden[hidden_layer <= 0] = 0
  dW = np.dot(X.T, dhidden)
  db = np.sum(dhidden, axis=0, keepdims=True)
  
  # We then add the regularization contributions
  dW2 += reg * W2
  dW += reg * W
  
  # And update the parameters
  W += -step_size * dW
  b += -step_size * db
  W2 += -step_size * dW2
  b2 += -step_size * db2

hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))

# We end by plotting the decision boundaries 

Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral,edgecolors='black')
plt.title("2-layer neural network decision boundaries", fontdict = {'fontsize' : 15})
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()