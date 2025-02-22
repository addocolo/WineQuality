# Wine Quality Prediction With Supervised Learning
## Introduction

In this study, we use a dataset from the University of California, Irvine, containing chemical features of different types of Vinho Verde from the north of Portugal. It also includes a quality score between 0 and 10. The data is divided into two files: one for red wines and one for white wines. For this study, we combine the two and attempt to classify the "high" quality wines from the "low" quality wines.

We will use various supervised learning models to find the best model to predict “high” and “low” quality wines based on their chemical features.

The main steps of our analysis include:

1.	Data Preparation: Cleaning and preprocessing the data to ensure its quality and consistency.

2.	Exploratory Data Analysis: Visualizing and understanding the distribution and relationships of the features.

3.	Baseline Modeling: Implementing and evaluating initial models using various supervised machine learning algorithms with default hyperparameters.

4.	Hyperparameter Tuning: Optimizing the models to improve their performance.

5.	Results: Comparing the different models used and analyzing the results of the best model found.

## Data: Preparation and analysis

### Importing and cleaning data

The initial step involves importing the data and checking for any duplicates or invalid entries. The dataset is composed of 3,961 white wines and 1,359 red wines. Each dataset has identical columns, which facilitates merging them into a single dataframe. Before the merge, we add a binary 'type' column, where 0 represents red wine and 1 represents white wine, allowing us to differentiate between the two types after merging. After removing duplicate entries, we combine the two dataframes into one unified dataset. To enhance readability, we move the 'quality' label to the end of the dataframe.


### Exploratory data analysis

For our EDA we will examine several visualizations of our data.

1. **Bar chart of wine qualities stacked**: (fig. 1.1) The bar chart illustrates the distribution of wine qualities across both red and white wines. Quality scores range from 3 to 9, with scores of 5 and 6 being the most frequent and representing the majority of the data. Conversely, there are relatively few quality scores of 3 and 9, which represent outliers of very low and very high quality, respectively. These trends persist across both red and white wines.

![1 - bar chart quality](https://github.com/user-attachments/assets/cd29703b-2f80-4d2a-9b89-c589cdcc6e16)

2. **Box Plot of Wine Feature Distributions**: (fig. 1.2) The box plot illustrates the distributions of various chemical features in the wine dataset. Several features, such as residual sugar and total sulfur dioxide, display right-skewed distributions with numerous outliers, indicating significant variability. In contrast, features like pH and density exhibit more normal distributions with fewer extreme values, suggesting more consistency. These visual summaries help identify potential outliers and understand the spread and skew of each feature, guiding further analysis and potential data transformations.

![2 - feature distributions](https://github.com/user-attachments/assets/4d2887d1-a0a4-48fa-b789-d12bd0d8a1cb)

3. **Feature Correlation Matrix**: (fig 1.3) The correlation matrix presents the relationships between various chemical properties and their impact on wine quality. Certain features like alcohol content have a moderate positive correlation with wine quality, suggesting that higher alcohol levels are associated with better quality wines. Conversely, features like volatile acidity show a negative correlation with quality, indicating that lower acidity is preferred. Perhaps unsurprisingly free sulfur dioxide and total sulfur dioxide are strongly correlated. Interestingly, it also identifies some features, such as volatiles acidity and total_sulfur dioxide, that are correlated with type, indicating how we might differentiate red from white by chemical features.

![3 - feature correlation](https://github.com/user-attachments/assets/3fd68baf-c56b-4085-9541-d754f410f054)

4. **Feature Pairwise Plot**: (fig 1.4) This plot helps visually expand on the correlation matrix. It provides a comprehensive view of the interaction between the features. 

![4 - feature pairwise](https://github.com/user-attachments/assets/96e3dd5b-a87a-418e-a552-36097b787321)

### Data preprocessing

To ensure consistent feature handling throughout our analysis, we first stored the feature names. We then addressed outliers and feature scaling. From our visualizations (fig. 1.3, 1.4), we identified highly skewed features—residual sugar, chlorides, and total sulfur dioxide—and applied a log transformation (log1p) to reduce skewness while preserving data distribution. For extreme values, we applied Winsorization (clipping at the 1st and 99th percentiles) to chlorides and free sulfur dioxide to prevent outliers from disproportionately influencing the models. Continuous numerical features were then normalized using MinMax scaling to ensure comparability across different ranges for logistic regression and K-neighbors.

Since we aimed to classify wines into two quality groups, we binarized the quality score by setting a threshold of 7: wines with a score of 7 or higher were labeled as high quality (1), while those below 7 were labeled as low quality (0). This threshold was chosen because it represents approximately 20% of the dataset, providing a reasonable proportion to be considered better than average. Finally, we split the dataset into training and test sets, allocating 80% for training and 20% for testing.

## Hyperparameter tuning

To optimize model performance, we conducted a grid search with cross-validation to explore different hyperparameter settings for each classifier. The grid search allows us to test multiple combinations of hyperparameters for each classifier and identify the best-performing configuration. We use K-fold cross-validation to reduce the change of overfitting to the training data.

We chose hyperparameter values based on a reasonable range around the default value for each parameter.

- **Logistic Regression**: Logistic regression is sensitive to the regularization parameter C, so a broad range is tested. We focus on l2 penalty, because it is compatible with a number of different solvers.

- **Random Forest Classifier**: We tested a number of different parameters to balance depth and complexity.

- **K-Neighbors Classifier**: We tested broad range of neighbors because it is the most critical hyperparameter, in addition to different metrics for calculating distance.

- **Gradient Boosting Classifier**: We tested multiple values of number of estimators and learning rates to find a balance between overfitting and underfitting.

Following this we insert the results of all models tested of each type into a dataframe in order to compare the performance across the different types of models.

## Results

In this section, we present the results of our analysis and model evaluation. We begin by comparing the performance of various models to identify the top-performing ones. This comparison is followed by an analysis of the best model, including its performance metrics, confusion matrix, and feature importance.

1. Model Performance Comparison
We evaluated several supervised learning models to classify wines as "high" or "low" quality based on their chemical features. Our goal was to determine which model offers the best predictive performance.

- **Top 10 Performing Models**: The table below lists the top 10 performing models, ranked by their mean test accuracy scores. This provides an overview of how each model performed during cross-validation.

- **Top Performing Model by Type**: To further explore model performance, we identified the top-performing model for each classifier type. The table below presents the best model of each type, along with their respective mean test accuracy scores and hyperparameters.

- **Box Plot of Mean Test Accuracy Scores**: We also visualized the distribution of mean test accuracy scores for all classifiers using a box plot. This helps to understand the variability in performance across different models and highlights the models that consistently achieve high accuracy.

2. Detailed Analysis of the Best Model
After identifying the best-performing model overall, we conducted a detailed analysis to evaluate its performance on the test set.

- **Best Model Evaluation**: The best model was selected based on its highest mean test accuracy score during cross-validation. We fitted this model with the optimal hyperparameters and evaluated its performance on the test set. The classification report and confusion matrix provide insights into the model's accuracy, precision, and recall scores.

- **Feature Importance**: Since the best model supports feature importance (Gradient Boosting / Random Forest), we analyzed the importance of each feature. This analysis helps to understand which features contribute most to the model's predictions and can offer insights into the underlying factors influencing wine quality.

- ## Conclusions

Model Results
One of the Gradient Boosting Classifier models emerged as the top-performing model during our testing, with a respectable accuracy score of 0.835. However, upon examining the box plot of performance of all models tested, the Random Forest Classifier appears to be a more consistent predictor, with a higher mean and floor. Overall, all models performed well in training, but due to its highest mean score and relative lack of outlying model scores, if we were to choose a model for this task, we would select the Random Forest Classifier over the others.

By far, the best individual feature for predicting quality appears to be alcohol content. This raises the question of whether people truly enjoy higher alcohol wines or if perhaps imbibing higher alcohol wine puts one in a more generous mood when assigning a rating. Though the answer is outside the scope of this study, we recommend interested readers test this out—responsibly and in moderation, of course.

Model Limitations
It is important to note that while the accuracy score is quite good, the model performs far better in predicting "low" quality wines than "high" quality wines. The top model's precision of 0.65 and recall of 0.43 for the "high" category are both rather unimpressive. This indicates that the model struggles to accurately identify high-quality wines, potentially missing a significant number of true high-quality samples (low recall) and incorrectly labeling low-quality wines as high-quality (moderate precision). It seems possible that these chemical features are inadequate for judging the lower end of the high-quality category and are mixing up wines that had an original score of 6 or 7. It's also possible that a more advanced model could discover underlying relationships that could better predict the quality.

## Conclusions

Model Results
One of the Gradient Boosting Classifier models emerged as the top-performing model during our testing, with a respectable accuracy score of 0.835. However, upon examining the box plot of performance of all models tested, the Random Forest Classifier appears to be a more consistent predictor, with a higher mean and floor. Overall, all models performed well in training, but due to its highest mean score and relative lack of outlying model scores, if we were to choose a model for this task, we would select the Random Forest Classifier over the others.

By far, the best individual feature for predicting quality appears to be alcohol content. This raises the question of whether people truly enjoy higher alcohol wines or if perhaps imbibing higher alcohol wine puts one in a more generous mood when assigning a rating. Though the answer is outside the scope of this study, we recommend interested readers test this out—responsibly and in moderation, of course.

Model Limitations
It is important to note that while the accuracy score is quite good, the model performs far better in predicting "low" quality wines than "high" quality wines. The top model's precision of 0.65 and recall of 0.43 for the "high" category are both rather unimpressive. This indicates that the model struggles to accurately identify high-quality wines, potentially missing a significant number of true high-quality samples (low recall) and incorrectly labeling low-quality wines as high-quality (moderate precision). It seems possible that these chemical features are inadequate for judging the lower end of the high-quality category and are mixing up wines that had an original score of 6 or 7. It's also possible that a more advanced model could discover underlying relationships that could better predict the quality.
