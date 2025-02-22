# WineQuality
A project on classifying into high and low quality based on chemical features.

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

2. **Box Plot of Wine Feature Distributions**: (fig. 1.2) The box plot illustrates the distributions of various chemical features in the wine dataset. Several features, such as residual sugar and total sulfur dioxide, display right-skewed distributions with numerous outliers, indicating significant variability. In contrast, features like pH and density exhibit more normal distributions with fewer extreme values, suggesting more consistency. These visual summaries help identify potential outliers and understand the spread and skew of each feature, guiding further analysis and potential data transformations.

3. **Feature Correlation Matrix**: (fig 1.3) The correlation matrix presents the relationships between various chemical properties and their impact on wine quality. Certain features like alcohol content have a moderate positive correlation with wine quality, suggesting that higher alcohol levels are associated with better quality wines. Conversely, features like volatile acidity show a negative correlation with quality, indicating that lower acidity is preferred. Perhaps unsurprisingly free sulfur dioxide and total sulfur dioxide are strongly correlated. Interestingly, it also identifies some features, such as volatiles acidity and total_sulfur dioxide, that are correlated with type, indicating how we might differentiate red from white by chemical features.

4. **Feature Pairwise Plot**: (fig 1.4) This plot helps visually expand on the correlation matrix. It provides a comprehensive view of the interaction between the features. 

### Data preprocessing

To ensure consistent feature handling throughout our analysis, we first stored the feature names. We then addressed outliers and feature scaling. From our visualizations (fig. 1.3, 1.4), we identified highly skewed features—residual sugar, chlorides, and total sulfur dioxide—and applied a log transformation (log1p) to reduce skewness while preserving data distribution. For extreme values, we applied Winsorization (clipping at the 1st and 99th percentiles) to chlorides and free sulfur dioxide to prevent outliers from disproportionately influencing the models. Continuous numerical features were then normalized using MinMax scaling to ensure comparability across different ranges for logistic regression and K-neighbors.

Since we aimed to classify wines into two quality groups, we binarized the quality score by setting a threshold of 7: wines with a score of 7 or higher were labeled as high quality (1), while those below 7 were labeled as low quality (0). This threshold was chosen because it represents approximately 20% of the dataset, providing a reasonable proportion to be considered better than average. Finally, we split the dataset into training and test sets, allocating 80% for training and 20% for testing.
