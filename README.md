# Simple-linear-reg-Delivery_time-prediction
Predict Delivery_time using sorting time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Data
data = {
    'Delivery Time': [21.00, 13.50, 19.75, 24.00, 29.00, 15.35, 19.00, 9.50, 17.90, 18.75, 19.83, 10.75, 16.68, 11.50, 12.03, 14.88, 13.75, 18.11, 8.00, 17.83, 21.50],
    'Sorting Time': [10, 4, 6, 9, 10, 6, 7, 3, 10, 9, 8, 4, 7, 3, 3, 4, 6, 7, 2, 7, 5]
}

df = pd.DataFrame(data)

# Summary statistics
print(df.describe())

# Univariate Analysis: Histogram
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Delivery Time'], bins=10, kde=True, color='skyblue')
plt.title('Delivery Time Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['Sorting Time'], bins=10, kde=True, color='salmon')
plt.title('Sorting Time Distribution')

plt.tight_layout()
plt.show()

# Bivariate Analysis: Scatter Plot
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='Sorting Time', y='Delivery Time')
plt.title('Delivery Time vs Sorting Time')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()

# Multivariate Analysis: Pairplot
sns.pairplot(df)
plt.suptitle('Pairplot of Delivery Time and Sorting Time', y=1.02)
plt.show()

# Linear Regression with Transformational Models
X = df[['Sorting Time']]
y = df['Delivery Time']

# Apply transformations
X_log = np.log(X)
X_square = np.square(X)
X_sqrt = np.sqrt(X)

# Transformations dictionary
transformations = {
    'Original': X,
    'Log': X_log,
    'Square': X_square,
    'Sqrt': X_sqrt
}

# Model fitting and evaluation for each transformation
for name, X_transformed in transformations.items():
    print(f"\nTransformation: {name}")
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE:', rmse)

    # Plot the regression line
    plt.figure(figsize=(7, 5))
    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_pred, color='red')
    plt.title(f'Linear Regression ({name}): Delivery Time Prediction')
    plt.xlabel('Sorting Time')
    plt.ylabel('Delivery Time')
    plt.show()
