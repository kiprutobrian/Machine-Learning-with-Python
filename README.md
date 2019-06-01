# Download and Install Python and SciPy Ecosystem
Start Python for the first time from command line by typing "python" at the command line.  Check the versions of everything you are going to need using the code below:

python3 -m pip install scipy

python3 -m pip install numpy

python3 -m pip install matplotlib

python3 -m pip install pandas

python3 -m pip install sklearn

# Basic-Python-and-SciPy-Syntax
Practice the basic syntax of the Python programming language and important SciPy data structures in the Python interactive environment.

Practice assignment, working with lists and flow control in Python.

Practice working with NumPy arrays.

Practice creating simple plots in Matplotlib.

Practice working with Pandas Series and DataFrames.

# Load Data and Standard Machine Learning Datasets
To get comfortable loading data into Python and to find and load standard machine learning datasets.

There are many excellent standard machine learning datasets in CSV format that you can download and practice with on the UCI machine learning repository.

Practice loading CSV files into Python using the CSV.reader() function in the standard library.

Practice loading CSV files using NumPy and the numpy.loadtxt() function.

Practice loading CSV files using Pandas and the pandas.read_csv() function.

# Understand Data with Descriptive Statistics.
To learn how to use descriptive statistics to understand your data. I recommend using the helper functions provided on the Pandas DataFrame.

Understand your data using the head() function to look at the first few rows.

Review the dimensions of your data with the shape property.

Look at the data types for each attribute with the dtypes property.

Review the distribution of your data with the describe() function.

Calculate pair-wise correlation between your variables using the corr() function.

# Understand Data with Visualization.
To improve your understanding of your data is by using data visualization techniques (e.g. plotting).

Today, your lesson is to learn how to use plotting in Python to understand attributes alone and their interactions. Again, I recommend using the helper functions provided on the Pandas DataFrame.

Use the hist() function to create a histogram of each attribute.

Use the plot(kind='box') function to create box and whisker plots of each attribute.

Use the pandas.scatter_matrix() function to create pair-wise scatter plots of all attributes.

# Prepare For Modeling by Pre-Processing Data.
The scikit-learn library provides two standard idioms for transforming data and each is useful in a different circumstance:

Fit and Multiple Transform.

Combined Fit-And-Transform.

There are many techniques that you can use to prepare your data for modeling. For example, try out some of the following:

Standardize numerical data (e.g. mean of 0 and standard deviation of 1) using the scale and center options.

Normalize numerical data (e.g. to a range of 0-1) using the range option.

Explore more advanced feature engineering such as Binarizing.

# Algorithm Evaluation With Resampling Methods.
The dataset used to train an algorithm cannot be used to give you reliable estimates of the accuracy of the model on new data. This is a big problem because the whole idea of creating the model is to make predictions on new data.

You can use statistical methods called resampling methods to split your training dataset up into subsets, some are used to 
train the model and others are held back and used to estimate the accuracy of the model on unseen data. 

Split a dataset into training and test sets.

Estimate the accuracy of an algorithm using k-fold cross validation.

Estimate the accuracy of an algorithm using leave one out cross validation.

# Algorithm Evaluation Metrics.
You can specify the metric used for your test harness in scikit-learn via the cross_val_score() function and defaults can be used for regression and classification problems. Your goal with today's lesson is to practice using the different algorithm performance metrics available in the scikit-learn package.

Practice using the Accuracy and Kappa metrics on a classification problem.

Practice generating a confusion matrix and a classification report.

Practice using RMSE and RSquared metrics on a regression problem.

# Spot-Check Machine Learning Algorithms.
You have to discover it using a process of trial and error. I call this spot-checking algorithms. The scikit-learn library provides an interface to many machine learning algorithms and tools to compare the estimated accuracy of those algorithms.

In this lesson, you must practice spot-checking different machine learning algorithms.

Spot-check linear algorithms on a dataset (e.g. linear regression, logistic regression and linear discriminate analysis).

Spot-check some nonlinear algorithms on a dataset (e.g. KNN, SVM and CART).

Spot-check some sophisticated ensemble algorithms on a dataset (e.g. random forest and stochastic gradient boosting).

# Model Comparison and Selection.
You will practice comparing the accuracy of machine learning algorithms in Python with scikit-learn.

Compare linear algorithms to each other on a dataset.

Compare nonlinear algorithms to each other on a dataset.

Create plots of the results comparing algorithms.

# Improve Accuracy with Algorithm Tuning.
One way to increase the performance of an algorithm is to tune it's parameters to your specific dataset.

The scikit-learn library provides two ways to search for combinations of parameters for a machine learning algorithm:

Tune the parameters of an algorithm using a grid search that you specify.

Tune the parameters of an algorithm using a random search.

# Improve Accuracy with Ensemble Predictions.
Some models provide this capability built-in such as random forest for bagging and stochastic gradient boosting for boosting. Another type of ensembling called voting can be used to combine the predictions from multiple different models together.

Practice using ensemble methods.

Practice bagging ensembles with the Random Forest and Extra Trees algorithms.

Practice boosting ensembles with the Gradient Boosting Machine and AdaBoost algorithms.

Practice voting ensembles using by combining the predictions from multiple models together.

# Finalize And Save Your Model.
You will practice the tasks related to finalizing your model.

Practice making predictions with your model on new data (data unseen during training and testing).

Practice saving trained models to file and loading them up again.

# Hello World End-to-End Project.
You need to practice putting the pieces together and work through a standard machine learning dataset end-to-end.
Work through the iris dataset end-to-end (the "hello world" of machine learning).

This includes the steps:

Understanding your data using descriptive statistics and visualization.

Pre-Processing the data to best expose the structure of the problem.

Spot-checking a number of algorithms using your own test harness.

Improving results using algorithm parameter tuning.

Improving results using ensemble methods.

Finalize the model ready for future use.
