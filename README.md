# Sweet Success: Analyzing Recipe Ratings with Data Science

## Recipe Analysis Project
**Author**: Timothy Kam  
**Date**: December 9, 2024  

---

# Introduction

Food is an integral part of daily life, not just for sustenance but also for enjoyment and expression. With the rising prominence of online recipe platforms like [Food.com](https://www.food.com), cooking has evolved from being a necessity to becoming an accessible and creative hobby. The digital age has allowed recipes to be rated, reviewed, and shared by millions, offering invaluable insights into public culinary preferences and trends.

This project aims to leverage data science to analyze and uncover patterns within recipes and user interactions on Food.com. Specifically, the research centers around the question:

**"What factors influence recipe preparation time, and how can we predict it based on recipe complexity and user interactions?"**

Understanding recipe preparation times is essential for everyday users planning their meals and for content creators optimizing their recipes for engagement and usability. By identifying trends and predictive factors, this study can help readers better understand the interplay between recipe characteristics and cooking time, making it a practical application of data science.

---

## The Dataset

This analysis uses two datasets sourced from Food.com, containing a wealth of information about recipes and user interactions. Data cleaning and feature engineering were performed to enhance the quality of the analysis.

### Recipes Dataset
This dataset contains **83,782 rows**, each representing a unique recipe, with the following key columns relevant to this study:

| Column               | Description                                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------------------|
| name              | Recipe name                                                                                                    |
| id                | Unique identifier for each recipe                                                                              |
| minutes           | Time (in minutes) to prepare the recipe                                                                        |
| n_steps           | Number of steps in the recipe                                                                                  |
| n_ingredients     | Number of ingredients required for the recipe                                                                  |
| tags              | List of Food.com tags associated with the recipe, such as "quick" or "vegetarian"                              |
| nutrition         | Nutrition details including calories, fat, sugar, protein, sodium, and carbohydrates                           |
| description       | A user-provided description of the recipe                                                                      |
| steps             | Text outlining the recipe steps                                                                               |
| ingredients       | Text listing the ingredients for the recipe                                                                    |

**New Features Created:**
- mean_ingredient_time: Average time spent on each ingredient, calculated as minutes / n_ingredients.
- ingredients_per_step: Average number of ingredients used per step, calculated as n_ingredients / n_steps.

---

### Interactions Dataset
This dataset contains **731,927 rows**, each capturing a user's interaction with a specific recipe. The key columns are:

| Column       | Description                                   |
|--------------|-----------------------------------------------|
| user_id    | Unique identifier for the user               |
| recipe_id  | Unique identifier linking to a recipe        |
| date       | Date when the interaction occurred           |
| rating     | Rating given by the user to the recipe (1–5 scale) |
| review     | Textual review provided by the user          |

---

## Data Cleaning and Exploration

### Data Cleaning Steps

To prepare the datasets for analysis, several data cleaning steps were undertaken:
1. **Merged Datasets**: The RAW_recipes.csv and RAW_interactions.csv datasets were merged using a left join on the id column from the recipes dataset and the recipe_id column from the interactions dataset. This ensured that all recipes, regardless of whether they had ratings, were retained.
2. **Handled Missing Values**: Ratings of 0 were replaced with NaN to avoid skewing the average ratings calculation, as a rating of 0 likely indicates missing data.
3. **Calculated Average Ratings**: The average rating for each recipe was computed and added as a new column, avg_rating, in the recipes dataset.

### Data Exploration

#### Distribution of Ratings
The majority of recipes receive ratings of 4 or 5, indicating a positive user bias.

**Visualization:**
<iframe
  src="assets/ratings_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### Distribution of Preparation Times
Most recipes take less than 60 minutes to prepare, with a sharp decline in frequency as preparation time increases.

**Visualization:**
<iframe
  src="assets/preparation_times.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### Distribution of Number of Ingredients
Recipes commonly use between 5 and 15 ingredients, with few exceeding 20.

**Visualization:**
<iframe
  src="assets/ingredients_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

## Relevance of the Research Question

The question about predicting recipe preparation times is practical and widely applicable:

1. **For Everyday Users**: Knowing the estimated time to prepare a recipe helps in meal planning, especially for busy individuals or families.  
2. **For Recipe Creators**: Identifying key factors influencing preparation time allows them to optimize recipes to suit specific audiences, like those seeking quick meals or more elaborate dishes.  
3. **For Data Enthusiasts**: This project showcases how exploratory data analysis, feature engineering, and predictive modeling can work together to address a real-world problem.

---

## Step 3: Assessment of Missingness

### NMAR Analysis

In this dataset, the review column could be classified as NMAR (Not Missing At Random). This is because whether a user leaves a review or not could depend on personal motivations, such as being highly satisfied or dissatisfied with a recipe. For instance, users who do not feel strongly about the recipe might be less likely to leave a review. Without additional data on user behavior or intent, it is difficult to establish that this missingness depends solely on observable data in the dataset, further supporting the NMAR classification.

### Missingness Dependency

To assess whether the missingness of rating is dependent on other variables, we conducted two separate permutation tests: one examining the relationship between the missingness of rating and review and another examining the relationship between the missingness of rating and description. For these tests, we utilized Pearson Correlation as the test statistic and ran 1000 permutations.

**Review and Rating**  
- Null Hypothesis: The missingness of rating does not depend on the missingness of review.  
- Alternative Hypothesis: The missingness of rating does depend on the missingness of review.  
- Test Statistic: Pearson Correlation  
- Significance Level: 0.05  
- Observed Statistic/P-Value: 0.167 (P-Value)

Since the p-value (0.167) is greater than the significance level of 0.05, we fail to reject the null hypothesis. This implies that the missingness of rating does not appear to depend on the missingness of review.

**Description and Rating**  
- Null Hypothesis: The missingness of rating does not depend on the missingness of description.  
- Alternative Hypothesis: The missingness of rating does depend on the missingness of description.  
- Test Statistic: Pearson Correlation  
- Significance Level: 0.05  
- Observed Statistic/P-Value: 0.521 (P-Value)

Similarly, the p-value (0.521) is greater than the significance level of 0.05, leading us to fail to reject the null hypothesis. This indicates that the missingness of rating does not appear to depend on the missingness of description.

### Visualizations

#### KDE Plot of Rating by Review Missingness
<iframe
  src="assets/rating_by_review.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### KDE Plot of Rating by Description Missingness
<iframe
  src="assets/rating_by_desc.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Conclusion

Based on the results of these permutation tests, there is no significant evidence to suggest that the missingness of rating is dependent on either review or description. This aligns with our earlier observation that the missingness in review might be NMAR, while the missingness in other columns could potentially be MAR (Missing At Random) or MCAR (Missing Completely At Random).

---

## Step 4: Hypothesis Testing

### Hypothesis Testing Overview

This analysis aims to determine whether the preparation time of a recipe influences its average rating. By dividing recipes into two categories—short and long preparation times—and comparing their ratings, we can identify any significant relationships between these variables.

**Null Hypothesis**: The preparation time of a recipe does not significantly affect its average rating.  
**Alternative Hypothesis**: The preparation time of a recipe significantly affects its average rating.  

### Methodology

1. **Data Categorization**: Recipes were divided into "Short Prep Time" and "Long Prep Time" categories based on the median preparation time.
2. **Test Statistic**: An independent t-test was performed to compare the mean ratings of the two categories.
3. **Significance Level**: A significance level of 0.05 was used for the test.

### Results

**Observed Statistic and P-Value**:  
- **T-Statistic**: 11.2012  
- **P-Value**: \(4.10 \times 10^{-29}\)  

### Descriptive Statistics

#### Short Preparation Time Recipes:
- **Count**: 115,959  
- **Mean**: 4.696  
- **Standard Deviation**: 0.684  

#### Long Preparation Time Recipes:
- **Count**: 103,434  
- **Mean**: 4.662  
- **Standard Deviation**: 0.739  

### Conclusion

Since the p-value is much smaller than the significance level of 0.05, we reject the null hypothesis. This suggests that the preparation time of a recipe significantly impacts its average rating. Specifically, users tend to rate recipes with shorter preparation times slightly higher on average. However, it is essential to interpret these findings cautiously, as the effect size, while statistically significant, is small.

---

### Visualization

**Ratings Distribution by Preparation Time Category**  
<iframe
  src="assets/ratings_by_prep_time_grouped.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The grouped bar chart above demonstrates that recipes with shorter preparation times tend to have a higher proportion of 5-star ratings compared to recipes with longer preparation times. This visualization supports the findings of our hypothesis test and highlights the subtle differences in user preferences for recipe preparation times.

---

## Step 5: Framing a Prediction Problem

### Prediction Problem
I aim to predict the **preparation time** (in minutes) for recipes based on their characteristics.

### Type of Prediction
This is a **regression problem** since we are predicting a continuous numerical value, which is the preparation time in minutes.

### Response Variable
The response variable is **minutes** (preparation time). I selected this variable because:
- It is an objective measure directly related to recipe characteristics.
- It has practical applications for users who want to plan their cooking time.
- It provides a meaningful target for regression modeling.

### Features Available at Prediction Time
I will only use features available **before cooking**. These include:
- **n_ingredients**: Number of ingredients in the recipe.
- **n_steps**: Number of steps in the recipe.
- **tags**: Tags that describe the recipe (e.g., cuisine type or dietary restrictions).
- **description_length**: The length of the recipe description in words.

I will **exclude** features that are unknown before cooking, such as:
- **User ratings**
- **Reviews**
- **User interactions**

This ensures the model aligns with real-world scenarios where preparation time must be estimated beforehand.

### Evaluation Metric
I will evaluate the model using **Root Mean Squared Error (RMSE)** because:
- It is a standard metric for regression problems.
- It measures the error in the same units as the target variable (minutes).
- It penalizes larger errors more heavily, which is important for time predictions.
- It is easy to interpret, as it provides an average measure of how far off the predictions are.

### Initial Analysis
From the earlier analysis:
- Preparation times range from 0 to 1,051,200 minutes (about 2 years).
- Most recipes take between 20–65 minutes (25th to 75th percentile).
- The median preparation time is 35 minutes, but there are significant outliers.

Correlations with recipe features are weak:
- **Number of Ingredients (n_ingredients)**: -0.008
- **Number of Steps (n_steps)**: 0.008

This suggests:
1. Outliers in the preparation time need to be handled.
2. The relationships between preparation time and recipe characteristics may be non-linear.

These insights will guide feature engineering and modeling choices.

---

## Step 6: Baseline Model

### Baseline Model Setup

The baseline model is a simple regression pipeline designed to establish a minimum performance benchmark for predicting preparation time.

**Features Used:**
- **n_ingredients**: Number of ingredients in the recipe.
- **n_steps**: Number of steps in the recipe.

**Pipeline Steps:**
1. **StandardScaler**: Standardizes features to have mean=0 and variance=1.
2. **LinearRegression**: Fits a simple linear model to predict preparation time.

### Data Filtering
To remove extreme outliers, only recipes with preparation times between 5 minutes and 24 hours were included in the dataset.

### Results

**Data Shape After Filtering:**  
- **Number of Recipes**: 227,027

**Training Metrics:**  
- **RMSE**: 96.15 minutes  
- **R² Score**: 0.0281

**Test Metrics:**  
- **RMSE**: 96.90 minutes  
- **R² Score**: 0.0285

**Feature Coefficients:**
- **n_ingredients**: +8.04 minutes (per additional ingredient).
- **n_steps**: +11.40 minutes (per additional step).

### Interpretation

Although the baseline model performs poorly (explaining only 2.85% of the variance in preparation time), it provides key insights:
1. Establishes a minimum performance benchmark.
2. Confirms intuitive relationships (e.g., more ingredients/steps correlate with longer preparation times).
3. Highlights the need for more sophisticated models and feature engineering to improve predictive performance.
