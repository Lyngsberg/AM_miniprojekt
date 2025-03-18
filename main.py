import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import variation

# Load UCI Wine Quality Dataset
path = 'data/dataset.csv'
data = pd.read_csv(path, delimiter=',')

# Features and target
X = data.drop(columns=['Quality']).values  # All features
y = data['Quality'].values  # Target (wine quality score)

# Split into initial training and pool set
X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.9, random_state=42)

# Initialize committee of regressors
committee_size = 5
committee = [RandomForestRegressor(n_estimators=10, random_state=i) for i in range(committee_size)]




# Train initial committee
for reg in committee:
    reg.fit(X_train, y_train)

# Evaluate initial performance
initial_predictions = np.mean([reg.predict(X_pool) for reg in committee], axis=0)
initial_mse = mean_squared_error(y_pool, initial_predictions)
print(f"Initial Model MSE: {initial_mse:.4f}")

# Function to calculate prediction variance (uncertainty measure)
def prediction_variance(committee, X):
    predictions = np.array([reg.predict(X) for reg in committee])
    return np.var(predictions, axis=0)  # Variance across committee predictions

committee_random = [RandomForestRegressor(n_estimators=10, random_state=i) for i in range(committee_size)]

# Select most uncertain samples
n_points = 500

uncertainty = prediction_variance(committee, X_pool)
most_uncertain_idx = np.argsort(uncertainty)[-n_points:]  # Select top 10 most uncertain samples

# Add selected samples to training set
X_train_al = np.vstack((X_train, X_pool[most_uncertain_idx]))
y_train_al = np.hstack((y_train, y_pool[most_uncertain_idx]))
X_pool_al = np.delete(X_pool, most_uncertain_idx, axis=0)
y_pool_al = np.delete(y_pool, most_uncertain_idx, axis=0)

idx_random = np.random.choice(len(X_pool), n_points)

# Add selected samples to training set
X_train_rand = np.vstack((X_train, X_pool[idx_random]))
y_train_rand = np.hstack((y_train, y_pool[idx_random]))
X_pool_rand = np.delete(X_pool, idx_random, axis=0)
y_pool_rand = np.delete(y_pool, idx_random, axis=0)

# Retrain committee
for reg in committee:
    reg.fit(X_train_al, y_train_al)

for reg in committee_random:
    reg.fit(X_train_rand, y_train_rand)

# Evaluate performance after retraining
final_predictions_al = np.mean([reg.predict(X_pool_al) for reg in committee], axis=0)
final_mse_al = mean_squared_error(y_pool_al, final_predictions_al)
print(f"Final Model MSE Active Machine Learning: {final_mse_al:.4f}")

# Evaluate performance after retraining
final_predictions_rand = np.mean([reg.predict(X_pool_rand) for reg in committee_random], axis=0)
final_mse_rand = mean_squared_error(y_pool_rand, final_predictions_rand)
print(f"Final Model MSE Random: {final_mse_rand:.4f}")

# Plot side-by-side histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot prediction variance distribution (Active Learning)
axes[0].hist(uncertainty, bins=30, edgecolor='k')
axes[0].set_xlabel('Prediction Variance')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Active Learning: Distribution of Prediction Variance')

# Plot histogram of random sample selection
random_variance = np.var([reg.predict(X_pool) for reg in committee_random], axis=0)
axes[1].hist(random_variance, bins=30, edgecolor='k')
axes[1].set_xlabel('Prediction Variance')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Random Sampling: Distribution of Prediction Variance')

plt.tight_layout()
plt.show()