import pandas as pd
from joblib import load
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


data = pd.read_csv('CarPrice_Assignment.csv')



print("\nFirst 5 rows of the dataset:\n")
print(data.head())


# encoding
categorical_cols = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Visualizing the car prices
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], bins=30, kde=True, color='blue')
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Predicting car price
X = data.drop(columns=['price', 'car_ID'])
y = data['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predicting and evaluating Linear Regression
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("Linear Regression Evaluation:")
print(f"  Mean Squared Error (MSE): {mse_linear:.2f}")
print(f"  R-squared (R²): {r2_linear * 100:.2f}%")


# Creating category
threshold = data['price'].median()
data['price_category'] = np.where(data['price'] > threshold, 1, 0)

X_class = data.drop(columns=['price', 'price_category', 'car_ID'])
y_class = data['price_category']


scaler = StandardScaler()
X_class_scaled = scaler.fit_transform(X_class)


X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class_scaled, y_class, test_size=0.2, random_state=42)


logistic_model = LogisticRegression(max_iter=2000, solver='lbfgs')
logistic_model.fit(X_train_class, y_train_class)

# Predicting and evaluating Logistic Regression
y_pred_class = logistic_model.predict(X_test_class)
accuracy_logistic = accuracy_score(y_test_class, y_pred_class) * 100  # Convert to percentage
classification_report_logistic = classification_report(y_test_class, y_pred_class)

print("\nLogistic Regression Evaluation:")
print(f"  Accuracy: {accuracy_logistic:.2f}%")
print("  Classification Report:\n", classification_report_logistic)

# Visualization for Logistic Regression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

conf_matrix = confusion_matrix(y_test_class, y_pred_class)
cmd = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Low", "High"])
cmd.plot(cmap=plt.cm.Blues)
plt.title("Logistic Regression: Confusion Matrix")
plt.show()

# Logistic Regression Predicting vs Actual Categories
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data.loc[y_test_class.index, 'price'], y=y_test_class, alpha=0.6, label='Actual', color='blue')
sns.scatterplot(x=data.loc[y_test_class.index, 'price'], y=y_pred_class, alpha=0.6, label='Predicted', color='orange')
plt.title("Logistic Regression: Predicted vs Actual Categories")
plt.xlabel("Price")
plt.ylabel("Category (0: Low, 1: High)")
plt.legend()
plt.show()

# to compare models
def compare_models():
    print("\nModel Comparison:")
    print("Linear Regression:")
    print(f"  MSE: {mse_linear:.2f}")
    print(f"  R-squared: {r2_linear * 100:.2f}%")
    print("\nLogistic Regression:")
    print(f"  Accuracy: {accuracy_logistic:.2f}%")
    print("  Classification Report:")
    print(classification_report_logistic)

compare_models()

# Predicting vs Actual Prices 
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_linear, alpha=0.7, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.title("Linear Regression: Predicted vs Actual Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()