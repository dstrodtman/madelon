# A Demonstration of the Benefits of Feature Selection Using the Canonical Madelon Dataset

This investigation into relevant feature identification utilizes two datasets of differing scales to test different methods for feature selection and modeling, and demonstrates the impact that proper feature engineering can have on the predictive strength of models. My results indicate in two datasets that these processes can improve predictive ability from just above chance to over 85% accuracy by leveraging machine learning principles and domain specific knowledge.

## The Data
#### UCI Data

The first dataset is the canonical Madelon dataset, maintained by and accessed from the University of California Irvine Machine Learning Repository <http://archive.ics.uci.edu/ml/datasets/madelon>. This dataset was previously used in the NIPS 2003 feature selection challenge, and so is well documented. A description provided by the UCI site:

>"MADELON is an artificial dataset containing data points grouped in 32 clusters placed on the vertices of a five dimensional hypercube and randomly labeled +1 or -1. The five dimensions constitute 5 informative features. 15 linear combinations of those features were added to form a set of 20 (redundant) informative features. Based on those 20 features one must separate the examples into the 2 classes (corresponding to the +-1 labels). We added a number of distractor feature called 'probes' having no predictive power. The order of the features and patterns were randomized."

2000 instances are provided in the training dataset, with 600 additional instances provided for validation/testing.

#### DSI Data

The second dataset was generated for the purposes of this project by a third party. The function utilized to create the data is based on the same math used to create Madelon, but the data varies in several ways, which will be discussed more below. Importantly, classes are assigned as 1 and 0, and I was naive to the specific parameters used to create the data (number of informative, redundant, and probe features). The feature space has been expanded to 1000, and 200k instances are included, which will be used to construct both the training and test sets.

Henceforth, these datasets will be referred to as UCI and DSI for brevity.

## UCI Data
### Benchmarking

Because of the massive size of DSI, UCI was used for early investigations into how to best approach the data. 3 samples of 600 instances were taken from the training data to match the size of test data.

Initial benchmark scores were fit for 4 models on each subset of the data. Additionally, because of the small size of the dataset, benchmarks were calculated using the entire training set. In all cases, data were scaled using StandardScaler from sklearn, which standardizes features by removing the mean and scaling to unit variance, and then passed to the respective classification model from sklearn using the default hyperparameters.

**Benchmarking Scores**

| Model			| Sample	| Score	|
| ---			| ---		| ---	|
| Logistic Regression 	| 1 		| 52.66	|
| Logistic Regression	| 2 		| 51.66 |
| Logistic Regression 	| 3		| 55.83 |
| Logistic Regression	| Full		| 57.99 |
| Decision Tree		| 1		| 66.66	| 
| Decision Tree		| 2 		| 66.16 |
| Decision Tree		| 3		| 63.33 |
Decision Tree | Full | 76.50
K Neighbors | 1 | 55.16
K Neighbors | 2 | 54.66
K Neighbors | 3 | 52.00
K Neighbors | Full | 50.66
Support Vector | 1 | 57.33
Support Vector | 2 | 55.16
Support Vector | 3 | 57.83
Support Vector | Full | 58.16

#### Logistic Regression

The LogisticRegression function from sklearn fits a regularized linear model to the data to make classification predictions. Prior knowledge about the multidimensional nature of the data suggests that this model should perform poorly, and benchmark scores confirm this. This provides a true benchmark with which to compare other data by essentially just drawing a line of best fit to the data to divide it into two classes.

#### Decision Tree Classifier

The DecisionTreeClassifier (DTC) function from sklearn predicts the value of a target variable by using the observed class of instances to derive simple rules about how the data functions. Because this model learns from the data to make its decisions, it should be expected to perform better on noisy data--it can ignore noisy features entirely, and makes decisions based on those features that seem to inform class. In all subsets, benchmark scores were higher than those from logistic regression, and with the full training set, a score of 76.5% is attained due to the increased number of instances from which the model can learn.

#### K Nearest Neighbors Classifier

The KNeighborsClassifier (KNC) function from sklearn stores the location and class of each observed instance and then uses these data to vote on the class of test points to predict class based on distance to previously observed points. With default parameters, results are only slightly above chance. This is be expected, as distances for class predictions are being measured in 500 dimensional space when the true function for the data is only 5 dimensions, and the signal is lost in the noise.

#### Support Vector Classifier

The SVC function from sklearn uses a radial basis function kernel (by default) to make predictions of class using a subset of training points (this subset comprises the support vector). While observed disparity between training and test scores are similar to those seen with the decision tree classifier, SVC does not benefit from increasing the sample size as decision trees did. It suffers from a similar issue as KNC--all features are maintained in the subset of instances used as support vectors, so noisy features greatly impact the outcome.

#### Summary

On the whole, the noise in our data prevents effective classification. The only model that can inherently eliminate noise in high dimensional space performs best, but by engineering features it is likely that all scores will improve.

### Identifying Features

Prior knowledge regarding the construction of the dataset provided the roadmap for a function to extract the 20 informative features out of the 500. While this exact technique will not generalize to most real-world datasets, it provides a nice benchmark against other methods for feature selection/reduction, and provides a good analogue to how domain knowledge can be leveraged alongside machine learning techniques to build better performing models.

Using Pearson correlation scores between all of the features, I was able to identify features that correlated with each other above 50%. Because the probe features are random noise, all of these features drop out with this simple filter. This method provided the same results on 3 samples of the data as it did on the full dataset.

By looking at the patterns of correlations between data, it is possible to determine that some of these features are repetitions of one another (the description of the data states that there are 5 true features, 5 redundant features, and 10 repetitions of these; I refer to these as 'feature bins').

