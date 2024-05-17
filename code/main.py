import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

'''
This first set of analysis is conducted by using tips as the target variable 
and it explores the relationship between the other pre-trip variables and 
tip amount. The data is scaled so that everything is normalized. 
'''

print("Exploratory Analysis With Tip Amount: ")

start_time1 = time.time()

# Ignore warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Load the dataset, file is stored locally in the same folder
file_path = 'fhvhv_tripdata_2022-07.parquet'
df = pd.read_parquet(file_path)

# Filter based on PULocationID, ID's are only for rides in Manhattan 
valid_location_ids = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263]
df_filtered = df[df['PULocationID'].isin(valid_location_ids)]

# Take a random sample
sample_df = df_filtered.sample(n=500000, random_state=42)

# Convert datetime columns to numeric
datetime_cols = ['request_datetime', 'on_scene_datetime', 'pickup_datetime']
for col in datetime_cols:
    sample_df[col] = pd.to_datetime(sample_df[col]).astype(int)

# Define the target and selected features, only pre-trip variables
selected_features = ['request_datetime', 'on_scene_datetime', 'pickup_datetime', 'trip_miles', 'sales_tax', 'congestion_surcharge', 'airport_fee', 'driver_pay', 'base_passenger_fare', 'trip_time', 'bcf', 'tolls']
y = sample_df['tips']
X = sample_df[selected_features]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and fit models, and calculate feature importances
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}
feature_importances = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_
    else:
        feature_importances[name] = np.abs(model.coef_)

# Calculate correlations
correlations = X.corrwith(sample_df['tips']).sort_values(ascending=False)

# ANOVA test for calculating p-values
p_values = {feature: f_oneway(df_filtered[df_filtered['tips'] == 0][feature],
                              df_filtered[df_filtered['tips'] > 0][feature]).pvalue
            for feature in selected_features}

# Print out results
print("Feature Importances:")
for model, importances in feature_importances.items():
    print(f"\n{model}:")
    for feature, importance in zip(selected_features, importances):
        print(f"{feature}: {importance}")

print("\nCorrelations with 'tips':")
print(correlations)

print("\nANOVA P-Values:")
for feature, p_value in p_values.items():
    print(f"{feature}: {p_value}")

# Visualizations
# Feature Importances for each model
for model, importances in feature_importances.items():
    plt.figure(figsize=(10, 5))
    sns.barplot(x=selected_features, y=importances)
    plt.title(f'Feature Importances ({model})')
    plt.xticks(rotation=45)
    plt.show()

# Heatmap of Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlations.to_frame('Correlation with tips'), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlations with Tips')
plt.show()

# Barplot of P-Values
plt.figure(figsize=(10, 5))
sns.barplot(x=list(p_values.keys()), y=list(p_values.values()))
plt.title('ANOVA P-Values')
plt.xticks(rotation=45)
plt.show()

end_time1 = time.time()
duration1 = end_time1 - start_time1

print(f"This program took {duration1} seconds to run.")

'''
This second set of analysis is similar to the ones above but our target variable
is now tip_flag. The value of tip_flag is either 1 - when tips is > 0.00
or 0 when tips is 0.00. This tests the relationship between the pre-trip
variables and whether or not there is a tip.
'''

print("Exploratory Analysis With Whether Riders Tip [tip_flag]: ")

start_time2 = time.time()

# Create a new column 'tip_flag' based on 'tips'
sample_df['tip_flag'] = np.where(sample_df['tips'] > 0.00, 1, 0)

# Define the target and selected features
selected_features = ['request_datetime', 'on_scene_datetime', 'pickup_datetime', 'trip_miles', 'sales_tax', 'congestion_surcharge', 'airport_fee', 'driver_pay', 'base_passenger_fare', 'trip_time', 'bcf', 'tolls']
target_variable = 'tip_flag'  # Using 'tip_flag' as the target variable
y = sample_df[target_variable]
X = sample_df[selected_features]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and fit models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"{name} - Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

# ANOVA test for calculating p-values
p_values = {feature: f_oneway(sample_df[sample_df['tip_flag'] == 0][feature],
                              sample_df[sample_df['tip_flag'] == 1][feature]).pvalue
            for feature in selected_features}

# Print out results
print("\nANOVA P-Values:")
for feature, p_value in p_values.items():
    print(f"{feature}: {p_value}")

# Visualizations
# Heatmap of Correlations
correlations = X.corrwith(sample_df['tip_flag']).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.heatmap(correlations.to_frame('Correlation with tip_flag'), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlations with tip_flag')
plt.show()

# Barplot of P-Values
plt.figure(figsize=(10, 5))
sns.barplot(x=list(p_values.keys()), y=list(p_values.values()))
plt.title('ANOVA P-Values')
plt.xticks(rotation=45)
plt.show()

end_time2 = time.time()
duration2 = end_time2 - start_time2

print(f"This program took {duration2} seconds to run.")


