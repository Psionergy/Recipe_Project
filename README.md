# Sweet Success: Analyzing Recipe Ratings with Data Science

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

#### Univariate Analysis
The distribution of ratings indicates a strong user preference for highly-rated recipes, with the majority of ratings concentrated at 4 and 5 stars. This reflects a generally positive user experience and suggests that users are more likely to leave ratings for recipes they enjoy.

**Visualization:**
<iframe
  src="assets/ratings_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### Distribution of Preparation Times
The distribution of preparation times shows that most recipes take less than 60 minutes to prepare, highlighting a user preference for efficiency in cooking. Longer preparation times are less frequent, suggesting that users value convenience and practicality when choosing recipes.

**Visualization:**
<iframe
  src="assets/preparation_times.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### Distribution of Number of Ingredients
The distribution of the number of ingredients reveals that recipes with 5 to 15 ingredients are the most common. This balance between variety and simplicity indicates that users are drawn to recipes that are neither overly complex nor too basic.

**Visualization:**
<iframe
  src="assets/ingredients_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

### Bivariate Analysis
For the bivariate analysis, we examined the relationships between pairs of columns to identify meaningful associations. First, we analyzed
the relationship between preparation time and the number of ingredients using a scatter plot. This plot shows that recipes with more 
ingredients tend to require more preparation time, though the correlation weakens as preparation time increases. Second, we explored how 
average ratings have changed over time using a line chart. This visualization reveals that recipe ratings have remained consistently high 
across years, with only slight fluctuations. These analyses provide insights into how different features of the recipes interact and inform 
potential hypotheses for further investigation. 

In addition to examining the scatterplot of preparation time against the number of ingredients, we explored the average preparation time 
for each number of ingredients. The resulting bar chart highlights a trend where recipes with fewer ingredients generally require less 
preparation time. However, there are noticeable spikes for recipes with either very few (1-2) or many ingredients (around 25), suggesting 
potential outliers or specific recipe types driving these averages. This analysis complements the scatterplot by providing aggregate-level 
insights into how the complexity of a recipe, as measured by its ingredients, correlates with the time required for preparation.

## Recipe Preparation Time vs. Number of Ingredients
<iframe
  src="assets/prep_time_vs_ingredients.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>



## Distribution of Recipe Ratings Over Time
<iframe
  src="assets/ratings_over_time.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>



## Average Preparation Time by Number of Ingredients
<iframe
  src="assets/ingredients_vs_prep_time.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>


## Relevance of the Research Question

The question about predicting recipe preparation times is practical and widely applicable:

1. **For Everyday Users**: Knowing the estimated time to prepare a recipe helps in meal planning, especially for busy individuals or families.  
2. **For Recipe Creators**: Identifying key factors influencing preparation time allows them to optimize recipes to suit specific audiences, like those seeking quick meals or more elaborate dishes.  
3. **For Data Enthusiasts**: This project showcases how exploratory data analysis, feature engineering, and predictive modeling can work together to address a real-world problem.


## Interesting Aggregates

This first table illustrates the relationship between preparation time and average recipe ratings. Recipes with the shortest preparation times (0-30 minutes) have the highest average rating (4.70), indicating a strong user preference for quick and simple recipes. As preparation times increase, average ratings gradually decline, with recipes requiring over 240 minutes receiving the lowest average rating (4.59). However, the variation across ranges is relatively small, suggesting that preparation time alone does not significantly influence user satisfaction.


### Preparation Time and Average Rating

| Preparation Time Range | Average Rating |
|-------------------------|----------------|
| 0-30                   | 4.701580       |
| 31-60                  | 4.672226       |
| 61-90                  | 4.679216       |
| 91-120                 | 4.670676       |
| 121-150                | 4.663806       |
| 151-180                | 4.688437       |
| 181-240                | 4.664046       |
| 241-300                | 4.594937       |


The second table examines how the number of steps in a recipe correlates with its average rating. Recipes with 0-5 steps have the highest average rating (4.71), highlighting a clear preference for straightforward recipes. Interestingly, recipes with 16-20 steps also receive relatively high ratings (4.68), suggesting that users value a balance between simplicity and detail. Despite minor fluctuations, ratings for recipes across step ranges remain close, indicating that the number of steps has only a marginal impact on user preferences.


### Steps and Average Rating

| Step Range | Average Rating |
|------------|----------------|
| 0-5        | 4.709929       |
| 6-10       | 4.671119       |
| 11-15      | 4.669430       |
| 16-20      | 4.684758       |
| 21-30      | 4.696182       |

These aggregates are interesting because they reveal user preferences related to recipe complexity and preparation effort. The preparation time analysis highlights a preference for quicker recipes, with ratings decreasing slightly as time increases, which can guide recipe creators aiming to cater to busy users. Similarly, the step-based analysis underscores a preference for simplicity, but also suggests that users appreciate well-structured recipes with moderate complexity, offering valuable insights into the balance between detail and ease of execution.

---

## Assessment of Missingness

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

## Hypothesis Testing

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
- **P-Value**: 0.00

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

## Framing a Prediction Problem

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

## Baseline Model

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

## Final Model

### Features Added and Their Relevance
In addition to the original features (`n_ingredients` and `n_steps`), two new features were engineered:
1. **`ingredients_per_step`**: This feature captures the average number of ingredients used per step. It provides a nuanced understanding of recipe complexity by combining the number of steps and ingredients into a single measure. Recipes with higher `ingredients_per_step` values are likely more intricate, potentially requiring more time per step.
2. **`calories`**: Extracted from the nutrition information column, calories act as a proxy for the recipe's complexity and richness. Recipes with higher calorie counts might involve more elaborate preparations, influencing preparation time.

These features were chosen because they represent aspects of the data generating process that directly relate to recipe preparation time. By accounting for both complexity and nutritional density, these features improve the model's ability to predict preparation time.

### Modeling Algorithm and Hyperparameter Selection
The final model used a **Random Forest Regressor**, a versatile and robust ensemble method. Random Forests were selected due to their ability to capture non-linear relationships and interactions between features, which were not captured by the baseline linear regression model.

To optimize the model, a **grid search** with cross-validation was performed, tuning the following hyperparameters:
- **`max_depth`**: Limited to 10 to prevent overfitting.
- **`min_samples_split`**: Set to 10 to ensure splits occur only when sufficient data exists.
- **`n_estimators`**: Fixed at 100 to balance computation time and performance.

These parameters were chosen to enhance generalization while preventing the model from memorizing the training data.

### Performance Comparison
The final model demonstrated a significant improvement over the baseline:
- **Test RMSE**: Reduced from 96.90 (baseline) to 89.13.
- **Test R² Score**: Increased from 0.0285 (baseline) to 0.1781.

These improvements highlight the added value of the engineered features and the non-linear modeling approach.

### Feature Importances
The importance of each feature in the final model provides further validation:
- **Calories** (59.28%): The most influential feature, indicating that recipe complexity related to nutritional density plays a significant role in preparation time.
- **Number of Steps** (16.98%): Suggests that more steps generally correlate with longer preparation times.
- **Ingredients per Step** (14.39%): Highlights how the interplay of ingredients and steps affects preparation time.
- **Number of Ingredients** (9.35%): Less influential when compared to the other features but still relevant.

### Conclusion
The Random Forest model, combined with engineered features, significantly improved prediction accuracy over the baseline. By incorporating domain-specific insights into feature engineering and leveraging an advanced modeling algorithm, the final model provides a robust tool for predicting recipe preparation times.

---

### Fairness Analysis

### Fairness Question
**Does our model perform worse for complex recipes compared to simple ones?**

#### Groups Analyzed
- **Group X (Simple Recipes)**: Recipes with 8 or fewer ingredients, representing simpler recipes.
- **Group Y (Complex Recipes)**: Recipes with more than 8 ingredients, representing more intricate and challenging recipes.

#### Evaluation Metric
We used **Root Mean Squared Error (RMSE)** as our evaluation metric. RMSE is particularly suited for regression problems as it measures the average magnitude of prediction errors in the same units as the target variable (minutes). A lower RMSE indicates better performance.

#### Hypotheses
- **Null Hypothesis**: The model is fair; any observed difference in RMSE between simple and complex recipes is due to random chance.  
- **Alternative Hypothesis**: The model is unfair; RMSE for complex recipes is significantly different from RMSE for simple recipes.

#### Results
The fairness analysis revealed the following metrics:
- **RMSE for Simple Recipes**: 84.11 minutes  
- **RMSE for Complex Recipes**: 95.61 minutes  
- **Observed Difference in RMSE**: 11.50 minutes  
- **P-value**: 0.0000  

The p-value was obtained using a permutation test with 1,000 iterations. By randomly shuffling the group labels and recalculating the RMSE difference for each permutation, we generated a distribution of RMSE differences under the null hypothesis. The observed difference (11.50 minutes) lies far outside this distribution, resulting in a p-value of approximately 0. This indicates that the observed difference is highly unlikely to have occurred due to random chance.

#### Interpretation
The extremely low p-value leads us to reject the null hypothesis in favor of the alternative hypothesis. This means that the model performs significantly worse for **complex recipes** compared to **simple recipes**. The higher RMSE for complex recipes suggests that the model struggles to capture the nuances and intricacies associated with more complex recipes, which often involve more ingredients and potentially non-linear interactions.

#### Importance of Findings
These results highlight a clear disparity in model performance between simple and complex recipes, which has practical implications:
1. **Model Limitations**: The findings suggest that the model is biased toward simpler recipes and may not generalize well to more complex ones. This could result from insufficient feature engineering or the model's inability to effectively capture complex relationships within the data.
2. **User Impact**: For users relying on the model to estimate preparation times for complex recipes, the increased error could lead to frustration or inaccurate expectations. This diminishes the model's usability and fairness for diverse recipe types.
3. **Future Improvements**: To address this bias, future iterations of the model should incorporate more advanced features or techniques, such as natural language processing to extract richer information from recipe descriptions and steps, or ensemble methods to better handle non-linear interactions.

#### Visualization
The histogram below illustrates the distribution of RMSE differences generated through the permutation test, with the observed difference (11.50 minutes) marked by vertical red lines. The observed difference lies far outside the distribution, reinforcing the conclusion of significant disparity.

**Visualization:**
<iframe
  src="assets/fairness_test.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---


