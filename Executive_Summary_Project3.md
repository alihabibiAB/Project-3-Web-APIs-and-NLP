
# Project 3: Web APIs and NLP

In this project, we test our knowledge about collecting text data using Pushift's API and perform binary classification on two subreddits.

Two subreddits selected for this project are: "Stocks" and "CryptoCurrency". These categories are very popular while they have some common words. Therefore overlaps make the classification harder than two categories without overlaps.

For more information about these two categories, you can check them out using the following links:

- [Subreddit category: "Stocks"](https://www.reddit.com/r/stocks/)

- [Subreddit category: "CryptoCurrency"](https://www.reddit.com/r/CryptoCurrency/)



## classification workflow

- [Importing Libraries](#import)

- [Collecting data using `request` library and Pushift's API](#API)

- [Performing EDA](#EDA)

- [Fitting and evaluating classification models(estimators)](#Fitting)

- [Summary](#summary)


<a id='import'></a>

## Importing libraries

Here, libraries required for classifications are imported. In addition to classifiers and sikit_learn libraries, `Wordcloud` and `eli5` libraries are imported to evaluate the feature importance analysis.  


<a id='API'></a>
## Collecting data using `request` library and Pushift's API

Pushift's API requests are prepared for two subreddits. Since Pushift's limitation is only 100 posts per request so several requests should be made to increase number of posts for generalized classifications. Here are steps performed for data collection:

1. Posts from "Stocks" and "CryptoCurrency" subreddits have been requested for a month starting from **Sunday, 21 March 2021 10:00:00** to **Tuesday, 20 April 2021 10:00:00 GMT**

2. Posts are requested for each category using the `message` function defined to request posts using Pushift's API.

3. Posts are converted into a dataframe and three features are extracted from them, including **subreddit, selftext, title**.

4. Two dataframes are concatenated to make the final dataframe. This data frame will be used for classification analysis.  

<a id='EDA'></a>
## Performing EDA


Now, the final dataframe composed of two subreddit posts is evaluated using EDA analysis. Steps taken for EDS analysis are:

1. The subreddit column is mapped in numeric values as **"CryptoCurrency":0** and **"Stocks":1**.

2. The number of null cells in **'title'** and **'selftext'** columns are counted and dropped from the dataframe. This is reasonable because there are only 27 null cells in **'selftext'** column.

3. The most frequent words in **'title'** and **'selftext'** column are visualized using the `WordCloud` library.

<a id=Fitting></a>
## Fitting and evaluating classification models(estimators)

Now, we can fit different classifiers to the final dataframe and evaluate their performances. Here are the steps taken to fit and evaluate a model:

1. **selftext** column is defined as **X** variable and **subreddit** column (having 0 and 1 values) is defined as **y** variable.
2. Training and test sets are split using `train_test_split` library with the test size of %30.
3. Before performing any hyperparameter and pipeline analysis, data type of **X** variable should be converted to Unicode or string. If datatype is  object, it would throw an error.
4. To evaluate a chain of transformers and estimators for different hyperparameters, `Pipeline()` and `GridSearchCV()` are used.
5. 7 pipelines are used to fit the training dataset and evaluate its performance on the test dataset.
6. The following table summarizes the main hyperparameters evaluated for each pipeline.

|Pipeline No.| Transformer|Transformer Hyperparameters|Estimator|Estimator Hyperparameters|
|:-------------------:|:--------------:|:-------:|:----------:|:-----------------:|
|1| `CountVectorizer()`|max_df=0.8, max_features=2000, min_df=3, stop_words='english'|`LogisticRegression()`|C=5, penalty='l1', solver='saga'|
|2|`TfidfVectorizer()`|max_df=0.8, max_features=2000, min_df=2, stop_words='english'|`LogisticRegression()`|C=10, penalty='none', solver='saga'|
|3|`TfidfVectorizer()`|max_df=0.8, max_features=100, min_df=3, stop_words='english'|`knn()`|n_neighbors=4|
|4|`TfidfVectorizer()`|max_df=0.8, max_features=100, min_df=2, ngram_range=(1, 2), stop_words='english'|`DecisionTreeClassifier()`|max_depth=7, min_samples_leaf=3,min_samples_split=3|
|5|`TfidfVectorizer()`|max_df=0.8, max_features=2000, min_df=2, stop_words='english'|`SVC()`|C=1|
|6|`TfidfVectorizer()`|max_df=0.8, max_features=100, min_df=2, stop_words='english'|`BernoulliNB()`||
|7|`TfidfVectorizer()`|max_df=0.8, max_features=1000, min_df=3,ngram_range=(1, 3), stop_words='english'|`RandomForestClassifier()`|n_estimators=200|


7. (i) Best training score based on CV, (ii) training score, (iii) test score, (iv) confusion matrix, and (v) feature importance parameter (`eli5` library) based on best estimator for each pipeline are calculated.   


<a id='summary'></a>
## Summary

1. **Important features based on the pipelines 1 and 2 (`LogisticRegression()`)**:
- Important features with positive and negative signs are related to Stocks and Cryptocurrency categories, respectively.
- Top five positive parameters in pipeline 1 are removed (+0.614), stock (+0.386), stocks (+0.362), company(+0.265), and shares(+0.217).

- Top five negative parameters in pipeline 1 are cryptos (-0.622), coins (-0.258), bitcoin(-0.245), binance(-0.209), and btc(-0.177).

- Top five positive parameters in pipeline 2 are relative (+0.614), started (+0.388), stay (+0.360), confirm (+0.265), and semi (+0.218).

- Top five negative parameters in pipeline 2 are break(-0.178), biden(-0.210), big(-0.246), coins(-0.260), and cuts(-0.623).



2. **Important features based on the pipeline 3 (`knn`)**:
- Eli5 library does not support the knn estimator. So, feature importance table can not be generated for this estimator.

3. **Important features based on the pipeline 4 (`DecisionTreeClassifier`)**:
- Top five parameters in pipeline 4 are stock (0.2946), removed (0.2724), stocks (0.1344), company (0.1035), and crypto (0.0462).

4. **Important features based on the pipeline 5 (`SVC`)**:

- Top five positive parameters in pipeline 5 are stock (+3.209), stocks (+3.046), company (+2.383), removed (+1.999), and shares (+1.998).
- Top five negative parameters in pipeline 5 are binance(-1.743), btc (-1.785), bitcoin (-1.982), coins (-2.162), and crypto (-4.372)

5. **Important features based on the pipeline 6 (Naive, `BernoulliNB`)**:
- Eli5 library does not support the Naive estimator. So, feature importance table can not be generated for this estimator.

6. **Important features based on the pipeline 7 (`RandomForestClassifier`)**:
- Top six parameters in pipeline 7 are removed (0.2855 ± 0.1299), crypto (0.0486 ± 0.1062), stock (0.0357 ± 0.0771), stocks (0.0297 ± 0.0619), company (0.0262 ± 0.0495), and coins (0.0147 ± 0.0379).


7. The following table summarizes the score values for different combination of transformaers and estimators.


|Pipeline No.| Transformer|Estimator|Best Score based on CV|Training Score|Test Score|
|:-------------------:|:--------------:|:-------:|:----------:|:-----------------:|:----------:|
|1| `CountVectorizer()`|`LogisticRegression()`|0.8112903225806452|0.8175115207373271|0.8032258064516129|
|2|`TfidfVectorizer()`|`LogisticRegression()`|0.8110599078341014|0.8258064516129032|0.8005376344086022|
|3|`TfidfVectorizer()`|`knn()`|0.7301843317972351|0.784331797235023|0.7559139784946236|
|4|`TfidfVectorizer()`|`DecisionTreeClassifier()`|0.7794930875576036|0.7861751152073733|0.7731182795698924|
|5|`TfidfVectorizer()`|`SVC()`|0.8131336405529954|0.8253456221198157|0.8026881720430108|
|6|`TfidfVectorizer()`|`BernoulliNB()`|0.7930875576036867|0.7976958525345622|0.7887096774193548|
|7|`TfidfVectorizer()`|`RandomForestClassifier()`|0.8110599078341014|0.8258064516129032|0.8016129032258065|

8. *LogisticRegression*, *SVM*, and *Random Forest* classifiers show the best score values. The best score values based on CV do not change significantly among these estimators. All three estimators have the best score value of 0.81. Interestingly, the score values obtained from these estimators for the test dataset are close to 0.8. This shows that the model is fitted properly. Moreover, the score values of the training dataset for these estimators are in the range 0.81-0.82, indicating that the over fitting issue does not exist.

9. Confusion matrices calculated for all estimators show that sensitivity values are in the range of 0.89-0.98 while specificity values are in approximately 0.63. This means that, the number of false negative values (wrongly predicted Crypto subreddit) is low while the number of false positive values (wrongly predicted stocks subreddit) is relatively high. One reason can be related to application of stocks words in crypto subreddit. However, application of crypto words in stocks is limited, leading to low false negative and high sensitivity value. 

10. High number of true positive and true negative values indicates that there are many unique words in these categories. Therefore, estimators can easily distinguish between these categories.  
