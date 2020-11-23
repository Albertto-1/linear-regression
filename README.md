# Linear Regression

## Objective
The purpose of this project is to document and exemplify the process of creating a [Linear Regression Model](https://en.wikipedia.org/wiki/Linear_regression#:~:text=In%20statistics%2C%20linear%20regression%20is,is%20called%20simple%20linear%20regression.) with the framework [scikit-learn](https://scikit-learn.org/stable/) and by hand.

## Requirements
To be able to run this project you need to install the requirements listed in requirements.txt. It is better to do it in a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#:~:text=virtualenv%20is%20used%20to%20manage,can%20install%20virtualenv%20using%20pip.).

```python
pip install -r requirements.txt
```

## Structure
|  File  |  Description  |
|---|---|
|  [extract.py](./extract.py)  |  Functions needed to extract the formated data from the datasets.  |
|  [myManualModel.py](./myManualModel.py)  |  Linear regression model implemented by hand.  |
|  [frameworkLR.py](./frameworkLR.py)  |  Linear regression using the framework.  |
|  [manualLR.py](./manualLR.py)  |  Linear regression by hand.  |
|  [test.py](./test.py)  |  Util file for some interesting scripting or data ploting.  |

The frameworkLR and manualLR scripts can use the carprice or the insurance dataset.

## Datasets
In the folder datasets/ you can find carprice.csv and insurance.csv. The extract script has the necesary functions to extract and transform the data from the .csv.
*  **[carprice](./datasets/carprice.csv)**:    Dataset of several car details and its price. Used in this project to predict the car price. From [Kaggle](https://www.kaggle.com/hellbuoy/car-price-prediction).

*  **[insurance](./datasets/insurance.csv)**:   Dataset of some individuals medical costs billed by health insurance in the US. Used in this project to predict the medical cost based on age, sex, bmi, children,smoker and region. From [Kaggle](https://www.kaggle.com/mirichoi0218/insurance).


## Process

### 1. Data analysis and transformation

The process of analysing and transforming the data can be found in this [Google Colab notebook](https://colab.research.google.com/drive/1fFKCOdDQsuNb6ke2-zwIkS1005RHJMxL?usp=sharing).

For more information about this process and categorical encoding for the carprices dataset click [here](https://pbpython.com/categorical-encoding.html).

### 2. Construction of [myManualModel.py](./myManualModel.py)

My manual Linear regression implementation is based in the *Amit Yadavs* course in [Coursera](https://www.coursera.org/learn/linear-regression/home/welcome).
This manual implementation is using [Gradient Descent (GD)](https://en.wikipedia.org/wiki/Gradient_descent#:~:text=Gradient%20descent%20is%20a%20first,function%20at%20the%20current%20point.) optimization method.

**Process:**
The first step was to initialize the model Weights (W) and the b. based on the number of params of the dataset. --> __init()__
For the W, I decided to initialize them with ones.
For the b, I decided to initialize it with one.
I decided this because this initialization was giving better results than random initialization.
For more information about initializing W click [here](https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78) or [here](https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html#initialize-weights).

Then I need to implement GD. The gradient descent algorithm can be simplified in 4 steps:

1. Get predictions for X with current values of W and b. --> __predict()__
2. Compute the loss between y and the predictions. --> __compute_loss()__
3. Find gradients of the loss with respect to parameters W and b. --> __calculate_gradients()__
4. Update the values of W and b by subtracting the gradient values obtained in the previous step. --> __update_W_and_b()__

Finally I need to implement the training loop. --> __train()__
I put an iteration limit in the train function and defined learning rate to be _5e-5_.

### 3. Implementing framework and manual models.

To this point I alrready have everything I need to implement this models.
The steps that both implementations follow are:
1. Get the respective data.
2. Split the data in _train_ and _test_.
3. Initialize the model.
4. Train the model with _train_ data.
5. Predict for _test_ data.
6. Show some real and predicted data.

## Conclusion
The sklearn LinearRegression model is better in any aspect (execution time, resources, prediction quality).
For learning purposes, it's really useful to implement this basic ML model. It helps to understand the algorithm, functionality and use cases of Linear Regression.
