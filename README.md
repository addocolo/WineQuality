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
# Import raw data
white_df = pd.read_csv('data/winequality-white.csv', sep=';')
red_df = pd.read_csv('data/winequality-red.csv', sep=';')
print("White:")
print(f"\tShape: {white_df.shape}")
print(f"\tDupes: {white_df.duplicated().sum()}")
print(f"\tNaN: {white_df.isna().sum().sum()}")
print()
print("Red:")
print(f"\tShape: {red_df.shape}")
print(f"\tDupes: {red_df.duplicated().sum()}")
print(f"\tNaN: {red_df.isna().sum().sum()}")
print()
print(f"Columns match: {white_df.columns.equals(red_df.columns)}")
print()
print("Columns:")
print(f"{white_df.columns.to_series()}")

white_df.head()
red_df.head()
# Add 'type' encoding
red_df['type'] = 0
white_df['type'] = 1

# Drop duplicates
red_df = red_df.drop_duplicates()
white_df = white_df.drop_duplicates()

# Merge into one wine list
df = pd.concat([white_df, red_df], ignore_index=True)

# Move 'quality' to last column of dataframe
quality = df.pop('quality')
df['quality'] = quality
df.describe()
### Exploratory data analysis

For our EDA we will examine several visualizations of our data.

1. **Bar chart of wine qualities stacked**: (fig. 1.1) The bar chart illustrates the distribution of wine qualities across both red and white wines. Quality scores range from 3 to 9, with scores of 5 and 6 being the most frequent and representing the majority of the data. Conversely, there are relatively few quality scores of 3 and 9, which represent outliers of very low and very high quality, respectively. These trends persist across both red and white wines.

2. **Box Plot of Wine Feature Distributions**: (fig. 1.2) The box plot illustrates the distributions of various chemical features in the wine dataset. Several features, such as residual sugar and total sulfur dioxide, display right-skewed distributions with numerous outliers, indicating significant variability. In contrast, features like pH and density exhibit more normal distributions with fewer extreme values, suggesting more consistency. These visual summaries help identify potential outliers and understand the spread and skew of each feature, guiding further analysis and potential data transformations.

3. **Feature Correlation Matrix**: (fig 1.3) The correlation matrix presents the relationships between various chemical properties and their impact on wine quality. Certain features like alcohol content have a moderate positive correlation with wine quality, suggesting that higher alcohol levels are associated with better quality wines. Conversely, features like volatile acidity show a negative correlation with quality, indicating that lower acidity is preferred. Perhaps unsurprisingly free sulfur dioxide and total sulfur dioxide are strongly correlated. Interestingly, it also identifies some features, such as volatiles acidity and total_sulfur dioxide, that are correlated with type, indicating how we might differentiate red from white by chemical features.

4. **Feature Pairwise Plot**: (fig 1.4) This plot helps visually expand on the correlation matrix. It provides a comprehensive view of the interaction between the features. 



# Bar chart of different wine qualities
pivot_df = df.pivot_table(index='quality', columns='type', aggfunc='size', fill_value=0)
pivot_df.plot(kind='bar', stacked=True, color=['red', 'yellow'], alpha=0.7, width=0.8, edgecolor='black')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.legend(['Red Wine', 'White Wine'])
plt.title('Fig 1.1 Bar Chart of Wine Qualities Stacked')
plt.show()

# Box plots of the feature distributions
columns = df.drop(columns=['type']).columns
fig = plt.figure(figsize = (18, 10))
fig.suptitle('Fig 1.2 Wine Feature Distributions', fontsize=18)
plt.subplots_adjust(top=0.94)
for i, col in enumerate(columns):
    plt.subplot(3, 4, i+1)
    sns.boxplot(x = df[col])

# Correlation matrix to examine general correlation of features
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6.5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Fig 1.3 Feature Correlation Matrix')
plt.xticks(rotation=45, ha='right')
plt.show()

# Pairwise plot of each feature
sns.pairplot(df, height=1.7);
plt.suptitle('Fig 1.4 Pairwise plot of wine features', y=1.05, fontsize=18);
### Data preprocessing

To ensure consistent feature handling throughout our analysis, we first stored the feature names. We then addressed outliers and feature scaling. From our visualizations (fig. 1.3, 1.4), we identified highly skewed features—residual sugar, chlorides, and total sulfur dioxide—and applied a log transformation (log1p) to reduce skewness while preserving data distribution. For extreme values, we applied Winsorization (clipping at the 1st and 99th percentiles) to chlorides and free sulfur dioxide to prevent outliers from disproportionately influencing the models. Continuous numerical features were then normalized using MinMax scaling to ensure comparability across different ranges for logistic regression and K-neighbors.

Since we aimed to classify wines into two quality groups, we binarized the quality score by setting a threshold of 7: wines with a score of 7 or higher were labeled as high quality (1), while those below 7 were labeled as low quality (0). This threshold was chosen because it represents approximately 20% of the dataset, providing a reasonable proportion to be considered better than average. Finally, we split the dataset into training and test sets, allocating 80% for training and 20% for testing.
# Store feature names for easy retrieval later
features = df.drop(columns=['quality']).columns

df_scaled = df.drop(columns='quality').copy()

from scipy.stats import zscore
outlier_features = df.columns[(np.abs(zscore(df)) > 3).any(axis=0)]
outlier_features = [x for x in outlier_features if x not in ['type', 'quality']]

# Log transorm to reduce skew
skewed_features = ['residual sugar', 'chlorides', 'total sulfur dioxide']
df[skewed_features] = np.log1p(df[skewed_features])

# Winsorization to remove extreme outliers
outlier_features = ['chlorides', 'free sulfur dioxide']
df[outlier_features] = np.clip(df[outlier_features], df[outlier_features].quantile(0.01), df[outlier_features].quantile(0.99), axis=0)

# Normalize continuous/numerical features
scaler = MinMaxScaler()
numerical_features = df.drop(columns=['type', 'quality']).columns
df_scaled[numerical_features] = scaler.fit_transform(df_scaled[numerical_features])

X = df_scaled[features]
# Binarize quality: 1 = high, 0 = low
y = (df['quality'] >= 7).astype(int)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
