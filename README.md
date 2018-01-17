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


