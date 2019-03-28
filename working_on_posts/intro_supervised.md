# A Brief Introduction to Supervised Learning

<b>Summary:</b> In this post I will discuss the details of <b>supervised</b> machine learning and its applications. Code examples will be used for demonstration but the theory and background knowledge will be the main focus.

Supervised learning is the most common subbranch of machine learning today. Typically, new machine learning practitioners will begin their journey with supervised learning algorithms. Therefore, the first of this three post series will be about supervised learning.

<hr>

Supervised machine learning algorithms are designed to learn by example. The name "supervised" learning originates from the idea that training this type of algorithm is like having a teacher supervise the whole process.

When training a supervised learning algorithm, the training data will consist of inputs paired with the correct outputs. During training, the algorithm will search for patterns in the data that correlate with the desired outputs. After training, a supervised learning algorithm will take in new unseen inputs and will determine which label the new inputs will be classified as based on prior training data. The objective of a supervised learning model is to predict the correct label for newly presented input data. At its most basic form, a supervised learning algorithm can be written simply as:

<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;Y&space;=&space;f(x)" title="Y = f(x)" style='display: block; margin: auto;'>

Where *Y* is the predicted output that is determined by a mapping function that assigns a class to an input value *x*. The function used to connect input features to a predicted output is created by the machine learning model during training.

Supervised learning can be split into two subcategories: <b>Classification</b> and <b>regression</b>.

### Classification

<img src='a-i-dan.github.io/images/supervised_learning_post/supervised learning post.png?raw=true' style='margin: auto; display: block;'>

During training, a classification algorithm will be given data points with an assigned category. The job of a classification algorithm is to then take an input value and assign it a class, or category, that it fits into based on the training data provided.

The most common example of classification is determining if an email is spam or not. With two classes to choose from (spam, not spam), this problem is called a binary classification problem. The algorithm will be given training data with emails that are both spam and not spam. The model will find the features within the data that correlate to either class and create the mapping function mentioned earlier: *Y=f(x)*. Then when provided with an unseen email, the model will use this function to determine whether or not the email is spam.

Classification problems can be solved with a numerous amount of algorithms. It is up to you to figure out which one to use. Here are a few:
  * Linear Classifiers
  * Support Vector Machines
  * Decision Trees
  * K-Nearest Neighbor
  * Random Forest

### Regression

<img src='a-i-dan.github.io/images/supervised_learning_post/supervised learning post2.png?raw=true' style='margin: auto; display: block;'>

Regression is a predictive statistical process where the model attempts to find the important relationship between dependent and independent variables.  

There are many different types of regression algorithms. The three most common are listed below:
  * Linear Regression
  * Logistic Regression
  * Polynomial Regression
