
# MNIST Classification With Keras Part One

In this post I will be using Keras to apply deep learning techniques to classify handwritten digits. The process will go something like this:

Load in the dataset, prepare the dataset, create the model, train the model, then test the model.

## What Is The MNIST Dataset?

MNIST = Modified National Institute of Standards and Technology

The MNIST dataset... One of the most beloved machine learning datasets of all time. Often refered to as the "Hello World" of machine learning, <b>MNIST is a dataset of handwritten digits, ranging from 0 to 9, where each digit resides in a 28x28 pixel image</b>. The dataset consists of 60,000 training set examples and 10,000 test set examples. Each handwritten digit also comes with a corresponding label, specifying the number it is. 

### Before We Get Started...

Before we get started, you will want to install Matplotlib, and Keras. <b>Keras</b> is a high level python deep learning library that is able to run on top of TensorFlow and Theano. Keras helps with fast and easy experimentation, making deep learning easier than ever. <b>Matplotlib</b> is a python plotting library. We will be using it to view the handwritten digits.

To install each library, in the terminal type this command:

`pip install matplotlib`

`pip install keras`

Now that we have the libraries installed we can get started with looking at the dataset and creating our model.


## Import Libraries

First, we want to import the needed libraries that were installed earlier.


```python
import matplotlib.pyplot as plt
import keras
```

## Load In The MNIST Dataset


```python
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## Looking At The Data


```python
print(x_train[7]) # will be a three (can you see the number?)
```

    [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0  38  43 105 255 253 253 253
      253 253 174   6   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0  43 139 224 226 252 253 252 252 252
      252 252 252 158  14   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0 178 252 252 252 252 253 252 252 252
      252 252 252 252  59   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0 109 252 252 230 132 133 132 132 189
      252 252 252 252  59   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   4  29  29  24   0   0   0   0  14
      226 252 252 172   7   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  85
      243 252 252 144   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  88 189
      252 252 252  14   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  91 212 247 252
      252 252 204   9   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0  32 125 193 193 193 253 252 252 252
      238 102  28   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0  45 222 252 252 252 252 253 252 252 252
      177   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0  45 223 253 253 253 253 255 253 253 253
      253  74   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0  31 123  52  44  44  44  44 143 252
      252  74   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  15 252
      252  74   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  86 252
      252  74   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   5  75   9   0   0   0   0   0   0  98 242 252
      252  74   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0  61 183 252  29   0   0   0   0  18  92 239 252 252
      243  65   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0 208 252 252 147 134 134 134 134 203 253 252 252 188
       83   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0 208 252 252 252 252 252 252 252 252 253 230 153   8
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0  49 157 252 252 252 252 252 217 207 146  45   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   7 103 235 252 172 103  24   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]]



```python
plt.imshow(x_train[7], cmap='bone')

print('y value: ',y_train[7])
```

    y value:  3



![png](images/MNIST_BLOG_files/MNIST_BLOG_7_1.png)


## Normalize The Data


```python
# normalize data
x_train = keras.utils.normalize(x_train, axis = 1)
x_test = keras.utils.normalize(x_test, axis = 1)
```


```python
print(x_train[7])
```

    [[0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.05360584
      0.06300896 0.15315156 0.33722666 0.31837327 0.30126264 0.27010914
      0.25773558 0.34097592 0.25832406 0.01332988 0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.07218017 0.20063235 0.31599232
      0.33116335 0.36756375 0.33458175 0.31711488 0.30007188 0.26904151
      0.25671686 0.33962819 0.3741245  0.35102012 0.16491144 0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.29879233 0.36373635 0.35549136
      0.36926179 0.36756375 0.33458175 0.31711488 0.30007188 0.26904151
      0.25671686 0.33962819 0.3741245  0.55985487 0.69498391 0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.18296834 0.36373635 0.35549136
      0.33702465 0.19253339 0.17588685 0.16610779 0.15718051 0.20178114
      0.25671686 0.33962819 0.3741245  0.55985487 0.69498391 0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.00671443 0.04185855 0.04090972
      0.03516779 0.         0.         0.         0.         0.01494675
      0.2302302  0.33962819 0.3741245  0.38212317 0.08245572 0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.09074813
      0.2475484  0.33962819 0.3741245  0.31991707 0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.10478701 0.20178114
      0.25671686 0.33962819 0.3741245  0.03110305 0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.12034363 0.26677918 0.29411807 0.26904151
      0.25671686 0.33962819 0.30286269 0.01999482 0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.05371548 0.18042478 0.27226124
      0.28280764 0.28150716 0.33458175 0.31711488 0.30007188 0.26904151
      0.24245482 0.13746855 0.04156939 0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.09847652 0.37265111 0.36373635 0.35549136
      0.36926179 0.36756375 0.33458175 0.31711488 0.30007188 0.26904151
      0.18031304 0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.09847652 0.37432972 0.36517975 0.35690204
      0.37072711 0.36902233 0.33722666 0.31837327 0.30126264 0.27010914
      0.25773558 0.09973209 0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.05203687 0.17753798 0.07335536
      0.06447428 0.0641778  0.05818813 0.05536926 0.17027888 0.26904151
      0.25671686 0.09973209 0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.01786142 0.26904151
      0.25671686 0.09973209 0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.10240548 0.26904151
      0.25671686 0.09973209 0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.01161786 0.14427079 0.0196953  0.         0.         0.
      0.         0.         0.         0.12332245 0.28816427 0.26904151
      0.25671686 0.09973209 0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.20040434
      0.4252136  0.48474986 0.06346265 0.         0.         0.
      0.         0.02625455 0.12166609 0.30075577 0.30007188 0.26904151
      0.2475484  0.08760251 0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.68334595
      0.58554004 0.48474986 0.32168998 0.22493355 0.19341536 0.18903112
      0.19635349 0.29609302 0.33458175 0.31711488 0.30007188 0.20071351
      0.08455357 0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.68334595
      0.58554004 0.48474986 0.55146853 0.42300937 0.36373635 0.35549136
      0.36926179 0.36756375 0.33458175 0.28943024 0.1821865  0.008541
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.16098054
      0.36480074 0.48474986 0.55146853 0.42300937 0.36373635 0.35549136
      0.31797543 0.30192736 0.19307879 0.05662766 0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.016265   0.19813189 0.51426629 0.42300937 0.24826449 0.14530004
      0.03516779 0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.        ]]



```python
plt.imshow(x_train[7], cmap='bone')
```




    <matplotlib.image.AxesImage at 0xb325490f0>




![png](images/MNIST_BLOG_files/MNIST_BLOG_11_1.png)


## Creating The Model

Loss functions: https://keras.io/losses/

Activation Functions: https://keras.io/activations/

Optimizers: https://keras.io/optimizers/


```python
model = keras.models.Sequential()

model.add(keras.layers.Flatten())  
model.add(keras.layers.Dense(128, activation=tf.nn.relu))  
model.add(keras.layers.Dense(128, activation=tf.nn.relu))  
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))  
```

## Training The Model


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 3)
```

    Epoch 1/3
    60000/60000 [==============================] - 6s 94us/step - loss: 0.2667 - acc: 0.9229
    Epoch 2/3
    60000/60000 [==============================] - 5s 84us/step - loss: 0.1104 - acc: 0.9659
    Epoch 3/3
    60000/60000 [==============================] - 5s 85us/step - loss: 0.0744 - acc: 0.9770





    <keras.callbacks.History at 0xb3256b630>



## Conclusion



# MNIST Classification With Keras Part Two

### Playing With Functions

To read about softmax blah blah --> https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/

Adam: https://arxiv.org/abs/1412.6980v8


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 3)
```

    Epoch 1/3
    60000/60000 [==============================] - 7s 115us/step - loss: 0.0159 - acc: 0.9945 0s - loss: 0.0161 - acc: 0.99
    Epoch 2/3
    60000/60000 [==============================] - 5s 89us/step - loss: 0.0126 - acc: 0.9957
    Epoch 3/3
    60000/60000 [==============================] - 6s 94us/step - loss: 0.0123 - acc: 0.9956





    <keras.callbacks.History at 0xb36564780>



Adagrad: 
http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf


```python
model.compile(optimizer = 'adagrad',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 3)
```

    Epoch 1/3
    60000/60000 [==============================] - 6s 97us/step - loss: 0.0049 - acc: 0.9984
    Epoch 2/3
    60000/60000 [==============================] - 8s 133us/step - loss: 4.2605e-04 - acc: 1.0000
    Epoch 3/3
    60000/60000 [==============================] - 7s 111us/step - loss: 2.2515e-04 - acc: 1.0000





    <keras.callbacks.History at 0xb36564b00>



Adadelta: https://arxiv.org/abs/1212.5701


```python
model.compile(optimizer = 'adadelta',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 3)
```

    Epoch 1/3
    60000/60000 [==============================] - 8s 137us/step - loss: 1.9908e-04 - acc: 1.0000
    Epoch 2/3
    60000/60000 [==============================] - 5s 85us/step - loss: 1.1549e-04 - acc: 1.0000
    Epoch 3/3
    60000/60000 [==============================] - 5s 87us/step - loss: 7.7711e-05 - acc: 1.0000





    <keras.callbacks.History at 0xb3190fd68>




```python

```
