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

![Correlation heat map](https://github.com/dstrodtman/madelon/blob/master/images/feat_corr.png)
*Fig 1. This heatmap shows the correlation between the 20 features selected. The target is also included for reference, though it was not used for feature selection.*

![Histogram of important features](https://github.com/dstrodtman/madelon/blob/master/images/feat_hist.png)
*Fig 2. This figure shows the normalized histograms of the 20 relevant features from the UCI dataset. Most features show relatively normal or bimodal distribution.*

![Mean and STD of important features](https://github.com/dstrodtman/madelon/blob/master/images/feat_mean.png)
*Fig 3. This figure shows the means plotted with their standard deviations of the 20 informative UCI features. While the means fall within a fairly small range, the standard deviations vary greatly between features.*

#### Select K Best

The SelectKBest function from sklearn uses a linear model to test the individual effect of each feature on the target and then selects the number (K) of features input by the user. When specifying k=20, this model correctly identifies 13 of the 20 informative features, showing.

#### Select From Model

The SelectFromModel function from sklearn uses the feature importances returned by another model to make decisions about which features to keep. As DTC performed best in the benchmarks, this model is passed to SelectFromModel. With default setting, this method extracted 83 features, but importantly did extract 17 of the 20 informative features.

#### Summary

Combining the results of Select K Best and Select From Model reveals 14 overlapping features. Comparing this result to the informative features extracted with the correlation method shows that 13 of these features are informative. However, several of the feature bins are not captured by either of these feature selection models. While feature selection models do pretty well at reducing the feature space, ultimately domain knowledge should also be applied whenever possible to perform feature selection.

### Principal Component Analysis

Now that the features have been reduced to the 20 informative features and the true noise has been removed, it's possible for perform principle component analysis (PCA) to transform these features and further reduce dimensionality. Because the classes were assigned in 5 dimensional space and the redundant features are linear combinations of the 5 true features used to assign class, reducing the 20 features to 5 principal components should transform the data to the 5 most important dimensions.

Note: While I explored using a radial basis function kernel to perform PCA with the UCI data, this method does not scale well and so was not explored on the DSI data. Implementation of an incremental process for using a radial basis function kernel to perform PCA could potentially integrate into a K Neighbors Classifier model the benefits observed from this kernel transformation in the Support Vector Classifier models described below.

### Testing Model Pipelines

Now that the informative features have been identified and the features have been reduced, the data is ready to be passed back through pipelines to identify the best models and hyperparameters for modeling the data.

To find the best hyperparameters for approaching the full dataset, I ran each of the 3 samples taken from the data through a grid search with 5 fold cross validation, meaning that each of the 3 samples was used to simulate a training and test set of data 5 times for each of the hyperparameters searched. This method ensures that model selection is not biased by the test data and helps to identify the best likely hyperparameters that will generalize to the full UCI data and perform well when predicting unseen data. Data were run through all models used in benchmark testing.

**Testing Models Scores**

Model | Sample | Score
--- | --- | ---
Logistic Regression | 1 | 56.66
Logistic Regression | 2 | 56.99
Logistic Regression | 3 | 61.49
Decision Tree | 1 | 71.99
Decision Tree | 2 | 76.50
Decision Tree | 3 | 75.00
Decision Tree (no PCA) | 1 | 72.99
Decision Tree (no PCA) | 2 | 73.16
Decision Tree (no PCA) | 3 | 73.83
K Neighbors | 1 | 91.33
K Neighbors | 2 | 89.33
K Neighbors | 3 | 88.16
Support Vector | 1 | 88.50
Support Vector | 2 | 88.50
Support Vector | 3 | 89.00

#### Logistic Regression

The results for logistic regression hardly improved from benchmark testing. Because of the multidimensional nature of the data and the linear fit used by logistic regression, this is unlikely to improve. This model will be removed from further analyses.

#### Decision Tree Classifier

On the samples of the data, DTC performed slightly better than the benchmark scores. Because the best scores were obtained on the benchmark using the full dataset, this model will continue to be explored moving forward, as the low scores are likely a result of an insufficient sample size.

#### K Neighbors Classifier

KNC performed the best on the sample data after it had been transformed using PCA. This was my expected result because of the nature of the way in which the data was generated. With more instances in the training set, the score should continue to improve.

#### Support Vector Classifier

SVC also performed very well on the sample data, though slightly worse than KNC. This model will also be tested on the full dataset.

### Building Final UCI Model

SVC and KNC provided the two best models for predicting class in the UCI data. In a dataset of this size, both models converged relatively quickly, although I am inclined to continue to favor KNC because of the known nature of the data.

**Final Scores**

Model | Score
--- | ---
Decision Tree | 83.83
K Neighbors | 92.00
Support Vector | 93.16

## DSI Dataset
### Sampling and Benchmarking

Because of the enormous size of the DSI data, only 3% of the the total data was used in each sample. Benchmark results perform similarly to the samples from UCI.

**Benchmark Scores**

Model | Score
--- | ---
Logistic Regression | 54.86
Decision Tree | 63.13
K Neighbors | 55.20
Support Vector | 59.13

### Feature Selection

Because DSI was created with the same basic function as UCI, the function used to identify important features was once again applied. It was cross validated in all three samples.

![Correlation heat map](https://github.com/dstrodtman/madelon/blob/master/images/dsi_feat_corr.png)
*Fig 4. A correlation heatmap for the DSI data. Note that the many repetitions of patterns of correlation are not observed in this dataset.*

![Histogram of important features](https://github.com/dstrodtman/madelon/blob/master/images/dsi_feat_hist.png)
*Fig 5. While some of the important features in the DSI dataset show a slight deviation from a true Gaussian distribution, no bimodal distributions are present in the data.*

![Mean and STD of important features](https://github.com/dstrodtman/madelon/blob/master/images/dsi_feat_mean.png)
*Fig 6. The DSI data all demonstrate means very near zero and standard deviations suggesting that they were constructed using an algorithm based on the normal distribution.*

SelectKBest performed very well on this dataset, with each sample identifying 17 of 20 important features. Select from Model also found 17 of the 20 features in each sample, but returned over 270 features out of the original 1000. Importantly, while some features appear in all of the samples, other features are present in only 1 or 2--this varies both between samples with the sample selection model and between models. Again, while these methods do fairly well at identifying relevant features, it is important to leverage domain knowledge to ensure the best results.

### Testing Model Pipelines

All samples of the data were passed through a similar processing pipeline to that used with UCI, which included scaling and principal component analysis with 5 fold cross validation.

**Testing Models Scores**

Model | Sample | Score
--- | --- | ---
Logistic Regression | 1 | 60.33
Logistic Regression | 2 | 60.53
Logistic Regression | 3 | 60.53
Decision Tree | 1 | 71.39
Decision Tree | 2 | 71.99
Decision Tree | 3 | 70.53
Decision Tree (no PCA) | 1 | 72.73
Decision Tree (no PCA) | 2 | 72.33
Decision Tree (no PCA) | 3 | 73.33
K Neighbors | 1 | 79.93
K Neighbors | 2 | 80.80
K Neighbors | 3 | 81.20
Support Vector | 1 | 81.53
Support Vector | 2 | 81.99
Support Vector | 3 | 81.39

#### Logistic Regression

Logistic regression performed above benchmark after the feature selection and PCA, but, as expected, is not a good model for this data.

#### Decision Tree Classifier

While DTC improved over the benchmark, it performs rather poorly on this data, and will not be further investigated.

#### K Neighbors Classifier

KNC shows the greatest improvement from the benchmark, and again scores nearly equally to SVC.

#### Support Vector Classifier

SVC shows some of the highest scores overall. However, because of the size of the dataset, SVC is not the best choice to process the full dataset because of the computation time associated with fitting the model.

### Building Final DSI Model

I chose to do an additional cross-validated grid search on the full DSI dataset to find the best number of neighbors to use for KNC. Based on the previous results from the samples and the UCI data, I felt this was the only hyperparameter that was likely to need to be further tuned. This model performed with 86.3% accuracy.
