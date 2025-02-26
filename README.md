# Wine Quality Prediction With Supervised Learning

## Introduction

In this study, we use a dataset from the University of California, Irvine, containing chemical features of different types of Vinho Verde from the north of Portugal. It also includes a quality score between 0 and 10. The data is divided into two files: one for red wines and one for white wines. For this study, we combine the two and attempt to classify the "high" quality wines from the "low" quality wines.

The goal of this project is to categorize wines into either "high" or "low" quality categories based on their chemical features. We would like to discover which features are most associated with higher quality wines. We will use various supervised learning models to identify the best model for this task.

The main steps of our analysis include:

1.	Data Preparation: Cleaning and preprocessing the data to ensure its quality and consistency.

2.	Exploratory Data Analysis: Visualizing and understanding the distribution and relationships of the features.

3.	Baseline Modeling: Implementing and evaluating initial models using various supervised machine learning algorithms with default hyperparameters.

4.	Hyperparameter Tuning: Optimizing the models to improve their performance.

5.	Results: Comparing the different models used and analyzing the results of the best model found.

## Data: Preparation and analysis

### Importing and initial cleaning of data

The initial step involved importing the data and checking for any duplicates or invalid entries. The dataset is composed of 4,898 white wines and 1,599 red wines. There are 11 chemical features and one target: quality. The chemical features and all are numerical variables. The features are fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.

Upon initial inspection, we noticed several duplicate entries in both the red and white wine datasets, which were subsequently removed.

Each dataset has identical columns, facilitating their merge into a single dataframe. Before merging, we added a categorical binary 'type' column, where 0 represents red wine and 1 represents white wine, enabling us to differentiate between the two types post-merging.

To classify wines into two quality groups, we binarized the quality score by setting a threshold of 7: wines with a score of 7 or higher were labeled as high quality (1), while those below 7 were labeled as low quality (0). This threshold was chosen to represent approximately 20% of the dataset, providing a reasonable proportion to be considered better than average. 

An inspection of the mean and maximum values of our features revealed some extreme outliers in the data. For example, mean sulfur dioxide has a mean value of about 30, but its highest value is 289. Additionally, the feature chlorides exhibits considerable positive skew. Preliminary correlation analysis indicates potential collinearity among certain features. These issues will be addressed in the Data Preprocessing section.

### Exploratory data analysis

For our EDA we will examine several visualizations of our data.

1. **Bar chart of wine qualities stacked**: The bar chart illustrates the distribution of wine qualities across both red and white wines. Quality scores range from 3 to 9, with scores of 5 and 6 being the most frequent and representing the majority of the data. Conversely, there are relatively few quality scores of 3 and 9, which represent outliers of very low and very high quality, respectively. These trends persist across both red and white wines.

![image](https://github.com/user-attachments/assets/f1a4619d-bb6c-4646-a21c-d63c85a59611)

2. **Box Plot of Wine Feature Distributions**: The box plot illustrates the distributions of various chemical features in the wine dataset. Several features, such as residual sugar and total sulfur dioxide, display right-skewed distributions with numerous outliers, indicating significant variability. In contrast, features like pH and density exhibit more normal distributions with fewer extreme values, suggesting more consistency. These visual summaries help identify potential outliers and understand the spread and skew of each feature, guiding further analysis and potential data transformations.

![image](https://github.com/user-attachments/assets/e719143e-909b-420e-8965-9fc58b863bed)

3. **Feature Correlation Matrix**: The correlation matrix presents the relationships between various chemical properties and their impact on wine quality. Certain features like alcohol content have a moderate positive correlation with wine quality, suggesting that higher alcohol levels are associated with better quality wines. Conversely, features like volatile acidity show a negative correlation with quality, indicating that lower acidity is preferred. Perhaps unsurprisingly free sulfur dioxide and total sulfur dioxide are strongly correlated. Interestingly, it also identifies some features, such as volatiles acidity and total_sulfur dioxide, that are correlated with type, indicating how we might differentiate red from white by chemical features.

![image](https://github.com/user-attachments/assets/d712c9c6-c659-450e-8684-0e39ced15588)

4. **Feature Pairwise Plot**: This plot helps visually expand on the correlation matrix. It provides a comprehensive view of the interaction between the features. 

![image](https://github.com/user-attachments/assets/19bcc074-2db4-49bb-aab8-15bbdb4798a7)

### Further Cleaning and Preprocessing

From our correlation matrix we see that the 'free sulfur dioxide' and 'total sulfur dioxide' were quite strongly correlated (0.72) though not perfectly collinear. While this is a strong correlation, we believed that it was not strong enough to indicate collinearity and as such decided not to combine them. Doing so may slightly improve the Logistic Regression model, but we believe it could only work to the detriment of the other models since the random forest and gradient boosting models can handle multicollinearity better and benefit from having both features. Additionally, preserving both 'free sulfur dioxide' and 'total sulfur dioxide' allows us to retain more nuanced information in our dataset, which may be beneficial for interpreting the results.

From our Exploratory Data Analysis (EDA), it is clear that we have a potential issue with outliers in our data. As a result, we numerically identified features with extreme outliers or positive skew and applied a log transformation to them. This transformation helps reduce the impact of skewness, bringing the data closer to a normal distribution. Stabilizing the variance in this manner should improve the performance, especially for models like Logistic Regression and K-Nearest Neighbors, which are more susceptible to skewed distributions.

To further address extreme outliers, we applied Winsorization by clipping the data at the 1st and 99th percentiles. This technique ensures that extreme outliers do not disproportionately influence the models, leading to more robust predictions.

We then applied a scaler to the numerical features of the dataset. This step is particularly beneficial for the K-Neighbors model, as it ensures that all features are on the same scale, preventing any single feature from dominating the distance metric used in the algorithm.

Given that our data is unbalanced with approximately a 4:1 ratio, we applied a random undersampling function to our training dataset. This approach helps ensure that the model is trained on balanced data, avoiding bias towards the larger category and improving overall model performance on minority class predictions.

Finally, we split our data into training (75%) and testing (25%) sets.

## Baseline Model Analysis

To establish a performance benchmark, we trained four supervised classification models (Logistic Regression, Random Forest, K-Nearest Neighbors, and Gradient Boosting) using their default hyperparameters. Each model was fitted to the training data, and predictions were evaluated on both the training and test sets. We measured a number of performance metrics including accuracy, F1, and ROC-AUC to compare performance across models, identifying potential overfitting or underfitting. These baseline results serve as a reference point for further model tuning and optimization.

- **Logistic Regression**: A linear model that estimates probabilities using a logistic function. It is highly interpretabie and serves as a good baseline, but may encounter issues if relationships are not linear.
- **Random Forest Classifier**: An ensemble of decision trees that is resistant to overfitting. Random Forest is useful for handling non-linear relationships in the data and provides feature importance metrics, enhancing interpretability.
- **K-Nearest Neighbors (KNN)**: A distance-based model that classifies samples based on their nearest neighbors. It can be effective when the decision boundary is complex and non-linear.
- **Gradient Boosting Classifier**: A powerful ensemble method that sequentially corrects errors from previous models. It builds trees in a stage-wise manner, optimizing a loss function for each new tree.

Each model was chosen to offer a mix of linear, tree-based, and instance-based approaches, allowing us to compare different learning paradigms before refining the best-performing model.

All models had accuracy, F1, and ROC-AUC scores within the range of 0.74-0.77. Based on these baseline results, our initial test did not strongly favor any particular model over the others. Therefore, we proceeded to the refinement stage with all four models. Although K-Neighbors lagged slightly behind the other classifiers, the difference was not significant enough to eliminate it at this stage.

![baseline results](https://github.com/user-attachments/assets/07730800-2fd4-4fcb-ad7b-fa025f764f83)

## Hyperparameter tuning

To optimize model performance, we conducted a grid search with cross-validation to explore different hyperparameter settings for each classifier. The grid search allows us to test multiple combinations of hyperparameters for each classifier and identify the best-performing configuration. We use 3-fold cross-validation to reduce the chance of overfitting to the training data.

We chose hyperparameter values based on a reasonable range around the default value for each parameter.

- **Logistic Regression**: Logistic regression is sensitive to the regularization parameter C, so a broad range is tested. We focus on l2 penalty, because it is compatible with a number of different solvers.

- **Random Forest Classifier**: We tested a number of different parameters to balance depth and complexity.

- **K-Neighbors Classifier**: We tested broad range of neighbors because it is the most critical hyperparameter, in addition to different metrics for calculating distance.

- **Gradient Boosting Classifier**: We tested multiple values of number of estimators and learning rates to find a balance between overfitting and underfitting.

We chose the F1 score as the primary metric for evaluating model performance. The F1 score offers a balanced measure by considering both precision and recall, which is critical for accurately identifying high-quality wines while minimizing false positives.

Following this we inserted the results of all models tested of each type into a dataframe in order to compare the performance across the different types of models.

## Results

In this section, we present the results of our analysis and model evaluation. We begin by comparing the performance of various models to identify the top-performing ones. This comparison is followed by an analysis of the best model, including its performance metrics, confusion matrix, and feature importance.

1. **Model Performance Comparison**: We evaluated several supervised learning models to classify wines as "high" or "low" quality based on their chemical features. Our goal was to determine which model offers the best predictive performance.   

   - **Top Performing Model by Type**: To further explore model performance, we identified the top-performing model for each classifier type. The table below presents the best model of each type, along with their respective mean test accuracy scores and hyperparameters. The top perfoming model was a Gradient Boosting Classifier with parameters outlined in the chart below. This model performed adequately with a 3-fold mean F1 score of 0.772505. This suggests a good, but not ideal, balance between identifying both classes. The top Random Forest model was very competitive with our top model with an F1 score of 0.771026 as was K-Neighbors with an F1 score of 0.770214. The Logistic Regression models were also competitive but a little behind the other methods.

![5 - top models](https://github.com/user-attachments/assets/a3242346-4456-4276-a580-031795bcabaa)

   - **Box Plot of Mean Test Accuracy Scores**: We also visualized the distribution of mean test accuracy scores for all classifiers using a box plot. This helps to understand the variability in performance across different models and highlights the models that consistently achieve high accuracy. The plot reveals that across all tested parameters all four models have close mean accuracies ranging from approximately 0.75 to 0.76. The Logistic Regression classifier was the most consistent performer and had the highest mean across all models, although it's best models didn't perform as well as some other methods. The Random Forest classifier was also relatively consistent and seemed to be immune to the lower end performance of some other models, perhaps because it does a good job of avoiding overfitting.

![6 - model accuracies](https://github.com/user-attachments/assets/9804dcb3-2a15-455e-a6d3-937c92598198)

2. **Detailed Analysis of the Best Model**: After identifying the best-performing model overall, we conducted a detailed analysis to evaluate its performance on the test set.

   - **Best Model Evaluation**: The best model was selected based on its highest mean test F1 score during cross-validation. As above the model selected was a Gradient Boosting Classifier with parameters {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 150, 'subsample': 0.9}. We evaluated its performance on the test set. The classification report and confusion matrix provide insights into the model's accuracy, precision, and recall scores.
      - Precision: For class 0, precision is 0.84, indicating a high rate of correctly identified low-quality wines among all predicted low-quality wines. For class 1, precision is 0.73, which is a respectable performance in identifying high-quality wines.
      - Recall: For class 0, recall is 0.71, meaning 71% of actual low-quality wines were correctly classified, which is also respectable if lower than we would have liked. For class 1, recall is a strong 0.85, meaning 85% of actual high-quality wines were correctly classified.
      - F1 Scores: The F1 scores of 0.77 (class 0) and 0.79 (class 1) reflect a balanced performance, meaning this model is similarly effective at handling both classes.
      - Accuracy: The model's accuracy score of 0.78 indicates a good but far from ideal performance for classifying high and low quality wines.

![7a - top model class report](https://github.com/user-attachments/assets/95c3c3da-791a-49c1-a268-c0b8e24a491f)
   
   - **Confusion Matrix Evaluation**
      - True Low (187): The model correctly identified 174 low-quality wines.
      - True High (206): The model correctly identified 209 high-quality wines.
      - False Low (36): 89 high-quality wines were incorrectly classified as low-quality.
      - False High (76): 33 low-quality wines were incorrectly classified as high-quality.

      The confusion matrix shows a respectable performance, with high numbers of true High and true Low classifications. While there are some misclassifications, the overall distribution suggests that the model is adept at distinguishing between high and low-quality wines, especially given the balanced nature of the F1 scores.

![7 - top model confusion](https://github.com/user-attachments/assets/5b33083d-2efe-4210-b187-38727597089a)

   - **Feature Importance**: Since the best model supports feature importance (eg. Gradient Boosting / Random Forest), we analyzed the importance of each feature. This analysis helps to understand which features contribute most to the model's predictions and can offer insights into the underlying factors influencing wine quality. There were some interesting findings:
      - By far the feature with the highest importance was alcohol (0.296), indicating that higher alchohol content is strongly associated with higher quality wines.
      - Volatile acidity and chlorides were second and third place respectively, although not nearly as important as alcohol.
      - The type, red or white, didn't seem to be of much importance at all (0.00077), suggesting that chemical features may vary in a similar way across classes regardless of the wine type.
![8a - feature importance](https://github.com/user-attachments/assets/560a4ebe-b6dc-4d70-ab8a-b716c328bddf)
![8 - top model feature importance](https://github.com/user-attachments/assets/f95d115a-d8fe-4adb-b5df-89418d1237b2)

## Conclusion

### Model Results

This study aimed to predict wine quality based on chemical features using various supervised learning models. While initial attempts to predict quality scores ranging from 0-10 yielded poor results, binarizing the quality scores provided a more manageable challenge.

Ultimately, the Gradient Boosting Classifier emerged as the top-performing model with a respectable F1 score of 0.78. However, when considering overall consistency and reliability, the Random Forest Classifier and Logistic Regression models also showed strong performance with more consistency. It's possible that over repeated random trials one or both of these models would emerge as winner. The Logistic Regression model could potentially be further improved by investigating feature interactions, which we did not explore in this study.

Interestingly, alcohol content was found to be the most significant predictor of wine quality, raising questions about whether higher alcohol content directly influences perceived quality or affects the ratings given by tasters. While this question lies outside the scope of our study, it provides an intriguing avenue for future exploration.

### Challenges with Data

Several challenges presented themselves during this study. Initially, we believed predicting wine quality would be straightforward, but it became clear that the task was more complex than anticipated.

One key challenge was the distribution of the quality scores, which led us to binarize the quality ratings to balance the classes better. Even then, creating this definition of "high" quality created an imbalance in the dataset. This issue was mitigated by undersampling our "low" quality data, but this may not be an ideal solution, as it results in a loss of information.

Additionally, dealing with outliers and skewed data was essential. Several chemical features had extreme outliers, which required careful preprocessing to avoid their disproportionate influence on the models. To address extreme outliers, we Winsorized our data. While this resulted in better predictions, it also led to a loss of information.

### Model Limitations
Despite our best efforts to overcome these challenges, the resulting models were only moderate predictors of wine quality. The chemical features used in this study may be inadequate for distinguishing between wines in the lower end of the high-quality category, possibly mixing wines that originally scored 6 or 7.

Moreover, it's possible that a more advanced model could uncover underlying relationships that our current models missed. The limitations of our approach highlight the need for further refinement and exploration.

### Future Study
To improve model performance, several potential avenues for future research could be explored. Analyzing interactions between features, particularly for Logistic Regression, might enhance model performance by providing deeper insights into feature relationships. Additionally, exploring more complex models, such as neural networks, could reveal nuanced patterns in the data that simpler models might miss. Using Principal Component Analysis (PCA) to reduce dimensionality could help capture the most relevant information while mitigating the impact of less informative features. Finally, including other relevant features not available in the current dataset might further improve predictions and refine the model's accuracy.

## References


Brownlee, Jason. (Mar. 17, 2021). SMOTE for Imbalanced Classification with Python. https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T.

Feature Importance with Random Forests. (05 Apr, 2024)
https://www.geeksforgeeks.org/feature-importance-with-random-forests/

Lundy, Daniel. (Jul 9, 2023). A Practical Guide to Wine Quality Prediction using Logistic Regression. https://medium.com/@daniel.lundy.analyst/a-practical-guide-to-wine-quality-prediction-using-logistic-regression-f390c5c4d71f.

Yagci, Hasan Ersan. (Jan 15, 2021). Detecting and Handling Outliers with Pandas. https://hersanyagci.medium.com/detecting-and-handling-outliers-with-pandas-7adbfcd5cad8
