import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import entropy

# Load dataset
data = load_wine()
X, y = data.data, data.target

# Split into initial training and pool set
X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.9, random_state=42)

# Initialize committee of classifiers
committee_size = 5
committee = [RandomForestClassifier(n_estimators=10, random_state=i) for i in range(committee_size)]

# Train initial committee
for clf in committee:
    clf.fit(X_train, y_train)

# Evaluate initial accuracy
initial_predictions = np.mean([clf.predict(X_pool) for clf in committee], axis=0).round().astype(int)
initial_accuracy = accuracy_score(y_pool, initial_predictions)
print(f"Initial Model Accuracy: {initial_accuracy:.4f}")

# Function to calculate vote entropy
def vote_entropy(committee, X):
    votes = np.array([clf.predict(X) for clf in committee])
    probs = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(np.unique(y)))/committee_size, axis=0, arr=votes)
    return entropy(probs, axis=0)

# Select most uncertain samples
uncertainty = vote_entropy(committee, X_pool)
most_uncertain_idx = np.argsort(uncertainty)[-10:]

# Add selected samples to training set
X_train = np.vstack((X_train, X_pool[most_uncertain_idx]))
y_train = np.hstack((y_train, y_pool[most_uncertain_idx]))
X_pool = np.delete(X_pool, most_uncertain_idx, axis=0)
y_pool = np.delete(y_pool, most_uncertain_idx, axis=0)

# Retrain committee
for clf in committee:
    clf.fit(X_train, y_train)

# Evaluate accuracy after retraining
final_predictions = np.mean([clf.predict(X_pool) for clf in committee], axis=0).round().astype(int)
final_accuracy = accuracy_score(y_pool, final_predictions)
print(f"Final Model Accuracy: {final_accuracy:.4f}")

# Plot vote entropy distribution
plt.hist(uncertainty, bins=30, edgecolor='k')
plt.xlabel('Vote Entropy')
plt.ylabel('Frequency')
plt.title('Distribution of Vote Entropy in the Pool Set')
plt.show()
