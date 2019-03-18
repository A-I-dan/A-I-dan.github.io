# Privacy In Deep Learning

Successful state-of-the-art deep learning techniques require a massive collection of centralized training data, or data kept on one machine or location. While deep learning has shown unprecedented accuracy and success in a numerous amount of tasks, the common use of centralized training data restricts deep learning's applicability to fields where exposed data does not present privacy risks. This causes some fields such as healthcare to be limited in its benefits from deep learning.  

It is a widely known fact that to receive the best results from a deep learning model, the training datasets must be bigger with lots of variety. The more data to learn from, the better the results. A model that lacks access to larger datasets and variety will commonly be subject to overfitting and poor end results.      

In the field of medicine, deep learning has proven to be potentially life saving and vital to the field's progression. The problem arises with the highly private and sensitive information that medical institutions carry. Due to the high privacy expectations, the data can not be legally shared. Each institution is then forced to train models on only the data available to them: the patient data they own and not from other institutions. This results in models producing results that to not generalize and will perform poorly when presented with new data.

## Federated Learning

The goal of federated learning algorithms is to train a model on data kept among a large number of different participating devices, called <b>clients</b>. As mentioned earlier, training deep learning models requires data to be stored locally along with the model itself. In other words, the training data is brought to the deep learning model. Federated learning aims to bring the model to the training data.

The model will use a large amount of clients to pull from. Training data will still be stored locally on each device, but each device will download the model, perform computations on their local data, produce a small update for the model, then update the model and send it back to the central server. This enables each client to contribute to the global models training while still storing its data locally. Each client will only send back an update to the global model, where it will then be averaged with other clients updates.

![fed_learning_chart](a-i-dan.github.io/images/privacy in deep learning figures.png?raw=true)


## Secure Multi-Party Computation




## Differential Privacy
