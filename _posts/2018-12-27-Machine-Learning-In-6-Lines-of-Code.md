# Machine Learning in 6 Lines of Code


In this post I will be reviewing a simple machine learning (ML) model. Our data set for this model will be three species of iris flowers with their pedal and sepal measurements.

<hr>

<h3>Supervised Learning</h3>

This is an example of <b>supervised learning</b> because we have labeled data to measure the success of our models predictions. In other words, we have a dataset that has both our input measurements and also our desired output. We can train our model on the correctly labeled data, then test the model with only inputs. This way, when we want to measure our models accuracy, we can go back and see if our test data matches the predictions.

The goal of creating this ML model is to distinguish between the three species of iris flowers. The three species are <b>setosa</b>, <b>versicolor</b>, and <b>virginica</b>. We want our model to learn from the measurements of flower. We then want to give our model a set of new inputs and have it predict the correct output.

<h5>Let's get started...</h5>

Here is the code we will be working with:

<script src="https://gist.github.com/A-I-dan/c1852b9950c00850a4e59fa675646b9d.js"></script>


<b>Note</b>: For this model we will be using scikit-learn. Scikit-learn, in my opinion, is not the best for learning what goes on underneath the hood of ML models. While I usually prefer to understand the "under the hood" part of the model, for this example we will not be going over that for simplicity reasons.

<hr>

<h5>Now let's really get started...</h5>

I will start off by first visualizing the dataset we will be working with. Below is the code for the visual:

<script src="https://gist.github.com/A-I-dan/778ff4574d31430a4877912b9fcf8214.js"></script>

<b>Visual</b>:

<img src='https://github.com/A-I-dan/blog/blob/master/images/iris_dataset_plot.png?raw=true'>


The scatter matrix will show three colors, each representing a species:

<b>Blue</b>: Setosa, <b>Green</b>: Virginica, <b>Orange</b>: Versicolor.

Along both the x-axis and y-axis there will be four labels for our measurements. Sepal length, sepal width, petal width and petal length.

If you look at the plots, you will notice that some have more distinct grouping than others. Some plots have points that mix in together while some plots will show a clear difference between species.

<h5>Now the code...</h5>

I will start off by explaining the needed Python packages to make this model work.

Here are the packages you will need:

<script src="https://gist.github.com/A-I-dan/43dc749f03a5af805d88817dd774a3fa.js"></script>


<b>(Line 1)</b> The data for our model is coming from the first line in our code:
`from sklearn.datasets import load_iris`. We are pulling from the datasets and taking out `load_iris`. Later, I will show you how we bring the data into our model.

<b>(Line 2)</b> In the second line of our code, we have our split function that allows us to split our data into two groups. One group will consist of 75% of the data and will be our training set. Like the name suggests, we will be using the training set to train our model. The second group will consist of the remaining 25% of the data that we will later use to test our model. To later use this function, we will write `from sklearn.model_selection import train_test_split`.

<b>(Line 3)</b> In the third line, we write `from sklearn.neighbors import KNeighborsClassifier`. This is where the magic starts happening. For our model, our choice of a classification algorithm will be a <b>k-nearest neighbors classifier</b> (one of the simplest ML algorithms). This algorithm will help us make our predictions by finding the closest related data point to the unknown inputs we give it. For example, if we give it measurements to a new flower, it will find the most similar point within the training data and give it that label. It finds the "nearest neighbor" in the data.

<b>(Line 4)</b> In the code's fourth line, we have `import numpy as np`. NumPy is one of the most important Python packages to have and is one of the most commonly used. NumPy is a linear algebra library that is useful for its n-dimensional arrays for data storage.

<hr>

<h5>Now we start the fun part...</h5>


These next six lines of code are where we will create and test our model.

Here is the code:

<script src="https://gist.github.com/A-I-dan/79b8f87802bc13ad418447f4d6214112.js"></script>


<b>(Line 1)</b> `iris_dataset = load_iris()`

As mentioned earlier when explaining why we import `load_iris` in this line of our code: `from sklearn.datasets import load_iris`. This is the data for our model. Here, we are taking `load_iris` and setting it to `iris_dataset`. This will contain our data's values.


<b>(Line 2)</b>  `X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state = 0)`

  This is where we utilize the `train_test_split` function that I mentioned earlier as well. The `train_test_split` function will randomly select data points and split them among the two groups (One with 75% of the data and one with 25%).
