import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import variation
from scipy.stats import ttest_ind

# Load UCI Wine Quality Dataset
path = 'data/dataset.csv'
data = pd.read_csv(path, delimiter=',')

# Features and target
X = data.drop(columns=['Quality']).values  # All features
y = data['Quality'].values  # Target (wine quality score)

initial_MSEs = []
final_MSEs_rand = []
final_MSEs_al = []

for j in range(20):

    rng = np.random.default_rng(seed=j) 

    # Split into initial training and pool set
    X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.9, random_state=j)

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

    idx_random = rng.choice(len(X_pool), n_points, replace=False)

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

    initial_MSEs.append(initial_mse)
    final_MSEs_al.append(final_mse_al)
    final_MSEs_rand.append(final_mse_rand)

mean_MSEs_al = np.mean(final_MSEs_al)
mean_MSEs_rand = np.mean(final_MSEs_rand)

print(f'Mean final mse random: {mean_MSEs_rand}')
print(f'Mean final mse AL: {mean_MSEs_al}')

# Perform a t-test to compare the two distributions (Active Learning vs Random Sampling)
t_stat, p_value = ttest_ind(final_MSEs_al, final_MSEs_rand, equal_var=False)

# Print the results
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")


plt.figure()
plt.hist(final_MSEs_al)
plt.hist(final_MSEs_rand)
plt.show()
# Plot side-by-side histograms

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Compute means and standard errors for confidence intervals
mean_mse_al = np.mean(final_MSEs_al)
mean_mse_rand = np.mean(final_MSEs_rand)
std_error_al = np.std(final_MSEs_al) / np.sqrt(len(final_MSEs_al))
std_error_rand = np.std(final_MSEs_rand) / np.sqrt(len(final_MSEs_rand))

# Bar plot with confidence intervals
fig, ax = plt.subplots(figsize=(8, 5))

# Bar positions
labels = ['Active Learning', 'Random Sampling']
means = [mean_mse_al, mean_mse_rand]
errors = [std_error_al, std_error_rand]

# Plot bars with error bars (95% CI)
ax.bar(labels, means, yerr=np.array(errors) * 1.96, capsize=10, color=['blue', 'orange'])
ax.set_ylabel('Mean Squared Error (MSE)')
ax.set_title('Comparison of Active Learning vs Random Sampling')

plt.show()



