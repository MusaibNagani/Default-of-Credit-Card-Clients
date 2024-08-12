import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_excel('data.xls', header=1)

# Separate features and target variable
X = data.drop(['default payment next month'], axis=1)
y = data['default payment next month']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid to search over
param_grid = {'C': [0.1, 1],
              'kernel': ['linear', 'rbf', 'poly']}

# Create an SVM model
svm = SVC(random_state=42)

# Use GridSearchCV to search over the parameter grid
grid_search = GridSearchCV(svm, param_grid=param_grid, cv=5)
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
y_pred_svm = grid_search.predict(X_test)

# Calculate the accuracy of the model
acc_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Classification Accuracy:", acc_svm)

export_data = pd.DataFrame({
    'C': [p['C'] for p in grid_search.cv_results_["params"]],
    'kernel': [p['kernel'] for p in grid_search.cv_results_["params"]],
    'accuracy': grid_search.cv_results_["mean_test_score"]
})

export_data.to_csv('SVM.csv', index=False)
