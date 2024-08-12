import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_excel('data.xls', header=1)

# Drop ID and target variable
X = data.drop(['default payment next month', 'ID'], axis=1)
y = data['default payment next month']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid to search over
param_grid = {'n_estimators': [50, 100],
              'learning_rate': [0.1, 1],
              'estimator__max_depth': [1, 2]}

# Create a base decision tree model
base_model = DecisionTreeClassifier(random_state=42)

# Create an AdaBoost model with the base decision tree model
model = AdaBoostClassifier(estimator=base_model, random_state=42)

# Use GridSearchCV to search over the parameter grid
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Classification report for all hyperparameter combinations:")
for params, mean_score, std_score in zip(grid_search.cv_results_['params'],
                                         grid_search.cv_results_['mean_test_score'],
                                         grid_search.cv_results_['std_test_score']):
    print(f"Hyperparameters: {params}")
    print(f"Mean test score: {mean_score:.3f} (std: {std_score:.3f})")
    print("Classification report:")
    y_pred = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("=" * 80)

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters found:", grid_search.best_params_)

# Use the best model to make predictions on the test set
y_pred_model = grid_search.predict(X_test)

# Calculate the accuracy of the model
acc_model = accuracy_score(y_test, y_pred_model)
print("AdaBoost Classification Best Accuracy:", acc_model)

export_data = pd.DataFrame({
    'n_estimators': [p['n_estimators'] for p in grid_search.cv_results_["params"]],
    'learning_rate': [p['learning_rate'] for p in grid_search.cv_results_["params"]],
    'estimator__max_depth': [p['estimator__max_depth'] for p in grid_search.cv_results_["params"]],
    'accuracy': grid_search.cv_results_["mean_test_score"]
})

export_data.to_csv('Adaboost.csv', index=False)
