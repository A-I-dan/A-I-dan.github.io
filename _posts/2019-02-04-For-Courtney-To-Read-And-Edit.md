
# Simple Neural Networks With The Tanh Function

### Introduction

In this post I will be implementing the <b>tanh</b> function for a simple 2-layer neural network. I will try to go in depth explaining what is happening within the neural network, and why it magically works. This post is inspired by [Andrew Trask's](http://iamtrask.github.io) intuitive blog post about a simple neural network. While learning about machine learning and deep learning, Andrew Trask's blog has been one of my frequently used resources and I would recommend it to everyone trying to learn about this field. 

### What is a Neural Network?

Artificial neural networks are biology based algorithms that are inspired by, and designed to work like, the human brain. Similar to the human brain's neurons, artificial neural networks consits of artifical neurons, or nodes. These neurons are all connected, similar to the synapses in the brain. These artificial synapses are all weighted differently according to the strength of their influence to the next neuron. To start, the neural network example used in this blog post will start with randomized weights. Given time to learn, these weights will be adjusted to produce more accurate results. This is where <b>backpropagation</b> comes into play. Backpropagation goes back through the layers of the neural network and adjusts the weights according to their contribution to the error of the outputs. 

Neural networks require data to "learn" from, similar to the human brain. The more data there is to learn from, the better the results. 

Neural networks will typically consist of an input layer with one or more hidden layers and an output layer. The input layer takes in the data that it will learn from. This data then feeds forward to the next layer, called the hidden layer. The learning occurs in the hidden layer(s). Hidden layers apply a function to the input data, or previous hidden layers. In this blog post, the hidden layer will be applying the <b>tanh</b> function. The hidden layers will make the data usable to the output layer, then it will hand the data over to the output layer. 

### Simple Neural Network

Here is the code that we will be working with:


```python
import numpy as np

def tanh(x, deriv = False):
    if deriv == True:
        return (1 - (tanh(np.exp2(2) * x)))
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)

input_layer = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])            
outputs = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

syn0 = np.random.random((3, 1)) - 1

for iter in range(10000):
    hidden_layer = tanh(np.dot(input_layer, syn0))
    hidden_error = outputs - hidden_layer
    hidden_delta = hidden_error * tanh(hidden_layer, True)
    syn0 += np.dot(input_layer.T, hidden_delta)
```

<iframe src="https://trinket.io/embed/python3/e305ca1f54" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

## How Does it Work?

I will post snippets of the code and break down what is happening and why it is happening. I will start with the first line where we will import Numpy. Numpy is a linear algera library that we will be using frequently throughout the code.


```python
import numpy as np
```

<center>Now we will have to build our tanh function, or <b>hyperbolic tangent</b> function. The hyperbolic tangent is similar to the normal tangent. Going back to trigonometry, we know that the tangent of x is equal to sine of x over cosine of x, written as: </center> 

<img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;\frac{sin(x)}{cos(x)}" title="\frac{sin(x)}{cos(x)}" />



<center>Similarly, the hyperbolic tangent (tanh) of x is equal to the hyperbolic sine (sinh) of x over the hyperbolic cosine (cosh) of x, or:</center>

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\frac{sinh(x)}{cosh(x)}" title="\frac{sinh(x)}{cosh(x)}" />

<center>The cosh(x) is represented by this formula:</center> 

<img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;cosh(x)&space;=&space;\frac{e^{x}&plus;e^{-x}}{2}" title="cosh(x) = \frac{e^{x}+e^{-x}}{2}" />

<center>And the sinh(x) is represented by this formula:</center>

<img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;sinh(x)&space;=&space;\frac{e^{x}-e^{-x}}{2}" title="sinh(x) = \frac{e^{x}-e^{-x}}{2}" />

<center>This means that tanh is represented by either one of these two formulas:</center>

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\frac{e^{x}&space;-&space;e^{-x}}{e^{x}&space;&plus;&space;e^{-x}}" title="\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}" />

<br>

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\frac{e^{2x}-1}{e^{2x}&plus;1}" title="\frac{e^{2x}-1}{e^{2x}+1}" />

<br>

<center>Another important part of our neural network is the derivative, or slope, of the tanh function, which can be written as:</center>

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;1&space;-&space;tanh^{2}(x)" title="1 - tanh^{2}(x)" />

### Why The Tanh Activation Function?

 <img src='http://mathworld.wolfram.com/webMathematica/RealLinePlots.jsp?name=Tanh&xMin=-5&xMax=5&nt=1'>

The tanh function, common among feed forward neural networks, is a non-linear activation function which helps with generalization and fitting to data. The tanh function is similar to the commonly used <b>sigmoid function</b>, otherwise known as the <b>logistic function</b>. The tanh function shares a similar sigmoidal shape, or "S" shape, to the logistic function, but tanh ranges from -1 to 1 while the logistic function ranges from 0 to 1. This is part of the reason that the tanh function is prefered over the logistic function. The tanh function has data more centered around zero and therefore has stronger gradients. Tanh can also be negative, providing a greater area for improvement, whereas the logistic function cuts off at zero and is restricted compared to the tanh function.


## In The Code

### Define The Tanh Function

The equations for the tanh function and its derivative are above. Either tanh formula will work. In this example I will be using:
```python
return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
```
But if you wanted to switch things up you could use the other equation. This would be written like this:
```python
return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
```
If you want to shorten the code then you can also use Numpy's tanh function. While this is of course simpler, I will not be using it because I prefer the equations to be written out for this example. In this post I want to make understanding the inner workings of a neural network as intuitive as possible. I feel like using `numpy.tanh()` has a potential to take away from that experience. Feel free to use which ever option feels best for you. 

Like the tanh function that we wrote ourselves, Numpy's tanh function takes in an input array (x) and returns the corresponding tanh values. Numpy's tanh function can be used in the code like so:
```python
return np.tanh(x)
```

Now to put all those equations into a code format, we define our tanh function with two arguments. The first argument being "x", or the input that will take the place of "x" in our equation. The second argument being `deriv = False`. We will use the derivative of the tanh function later on, so we want to be able to tell our function whether it should use the derivative equation or not. If we call `deriv = True`, then the tanh function will return the derivative, but if we do not specify the derivative as true, the function will assume it to be false and will return the tanh function.

All together, the tanh function should look something like this:


```python
def tanh(x, deriv = False):
    if deriv == True:
        return (1 - (tanh(np.exp2(2) * x)))
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
  # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
  # return np.tanh(x)
```

### Create The Dataset

Now we will be creating our input and output matrices. Our input matrix will be called "input_layer" and our outputs will be appropriately names "outputs". Can you see the correlation between the inputs and the outputs? 

<b>Hint</b>: If the input data example starts with a zero, then the output will be a zero. If the input data example starts with a one, then the output will be a one as well.

We will be using numpy's arrays to help create our data. For the input data, each set of three numbers is a training example. Because there are three numbers in each training example, there are three <b>input nodes</b> in our neural network. Each training example has a corresponding output. Because the output is a single number (0 or 1), we have one output node in our neural network. 


```python
input_layer = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])            
outputs = np.array([[0, 0, 1, 1]]).T
```

At the end of the line where we create the outputs there is a "<b>.T</b>". This is the transpose function that will flip a matrix over its diagonal. In this case, the line of code creates a matrix with one row and four columns. Once the transpose function is applied, the outputs matrix will flip to a one column, four row matrix. That matches our four training examples, and one output node. Here is an example that prints out the outputs before and then after the transpose function is applied. Hopefully this helps understand what is going on.

```python
# TRANSPOSE
print('Before Transpose Function: ', outputs)
print('After Transpose Function: \n', outputs.T)
```
```
Before Transpose Function:  [[0 0 1 1]]
After Transpose Function: 
 [[0]
 [0]
 [1]
 [1]]
 ```

### Random Seed


```python
np.random.seed(1)
```

Seeding the random numbers means that even though the numbers will be generated randomly, it will be generated the same "random" way each time you run the code. To make this more intuitive or more confusing, imagine that ".seed(1)" was a predetermined set of "random" numbers. Then ".seed(2)" will have another set of random numbers, but different than ".seed(1)". Everytime you have ".seed(1)", you will get the same random set of numbers. This makes it easier to comapre results after running the code a few times. You can seed the numbers to any number that you want, but keep it the same each time you run it.

### Creating The Synapse


```python
syn0 = np.random.random((3, 1)) - 1
```

This is where we make the first and only artificial synapse for this 2-layer neural network. This will create a randomized weight matrix that has 3 inputs and one output that need to become connected. Everytime the code is run, the weights will be initialized with the same values. This is because we seeded our random numbers in the line of code above.

To view the random weights that the neural network will start with, you can simply print out the values in the synapse. Hit run in the code to view the weights that will be used in this neural network. Try playing around with the value in `np.random.seed(1)` to get different random weight values.

<iframe src="https://trinket.io/embed/python3/cf3a08193c" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

### Training


```python
for iter in range(10000):
    
    hidden_layer = tanh(np.dot(input_layer, syn0))
    
    hidden_error = outputs - hidden_layer
    
    hidden_delta = hidden_error * tanh(hidden_layer, True)
    
    syn0 += np.dot(input_layer.T, hidden_delta)
```

The training begins here. This <b>for loop</b> will run through this process however many times it is specified to do so. In this case we want to run through 10,000 training iterations. 


`hidden_layer = tanh(np.dot(input_layer, syn0))`

The hidden layer will be calculated by taking the product of the input dataset matrix and the synapse (weights) matrix, then running that result through the tanh function. In other words, we multiply the input matrix and the weights matrix together, then we make that result the "x" value in our tanh function. 


`hidden_error = outputs - hidden_layer`

The error will then be calculated by taking the difference of the correct outputs and the hidden layers prediction. 


`hidden_delta = hidden_error * tanh(hidden_layer, True)`

We will then do some more calculations for the "hidden_delta". First, notice that the `deriv = False` argument has now been set to "True". This means that our tanh function will now find the derivative of the hidden layer. This will then be multiplied by error that was calculated in the line above. This will help minimize the error for future iterations. 


`syn0 += np.dot(input_layer.T, hidden_delta)`



### Predicting

We can add one more step to test out the neural network that we have created. To predict the output to a new set of inputs we can create a "predict" function. This predict function will take in our new inputs and will spit out its predicted output. 

```python
def predict(new_input):
    return tanh(np.dot(new_input, syn0))
```

<b>Remember</b>: The first number in the training examples was what determined the output. Also, keep in mind that the training was done with 0s and 1s. Other numbers will obviously not work. 

Next we can create our "test dataset", or `test_data`, exactly like how we made our training data. Instead of writing the input data on one line, I wrote the test data on multiple lines to show the matrix it creates.

```python
test_data = np.array([ [1,1,0],
                       [0,1,0],
                       [1,0,0] ])
# or --> test_data = np.array([ [1,1,0], [0,1,0], [1,0,0] ])
```

After testing our neural network, we can loop through each test example and print out the predicted output. The neural network will not be 100% confident in its prediction and therefore will not say whether the output is 0 or 1. Instead, the neural network will spit out a number that is close to 1 or close to 0. The closer the number is to 1 or 0, the more confident the neural network is.

```python
for test_example in test_data:
        print('Predicted Output For:', test_example,'=', predict(test_example), '\n')
```

## All Together + Test Results


```python
import numpy as np

def tanh(x, deriv = False):
    if deriv == True:
        return (1 - (tanh(np.exp2(2) * x)))
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
    
input_layer = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])   

outputs = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

syn0 = np.random.random((3, 1)) - 1

for iter in range(10000):
    hidden_layer = tanh(np.dot(input_layer, syn0))

    hidden_error = outputs - hidden_layer
    
    hidden_delta = hidden_error * tanh(hidden_layer, True)

    syn0 += np.dot(input_layer.T, hidden_delta)
    
def predict(new_input):
    return tanh(np.dot(new_input, syn0))

test_data = np.array([ [1,1,0],
                       [0,1,0],
                       [1,0,0] ])

for test_example in test_data:
        print('Predicted Output For:', test_example,'=', predict(test_example), '\n')
```

    Predicted Output For: [1 1 0] = [0.99999874] 
    
    Predicted Output For: [0 1 0] = [-0.01759575] 
    
    Predicted Output For: [1 0 0] = [0.99999878] 
    


Try experimenting with the code below to help understand what is happening.

<iframe src="https://trinket.io/embed/python3/bc9e9ff5c8" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>
