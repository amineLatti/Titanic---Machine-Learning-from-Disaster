import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from preprocessing import preprocess_data

# Load training data
train_data = pd.read_csv('train.csv')

# Preprocess the training data
X, y = preprocess_data(train_data, is_train=True)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X.values, y.values.reshape(-1, 1), test_size=0.2, random_state=42)

# Train the Decision Tree model
tree = DecisionTree(min_samples_split=5, max_depth=6)
tree.fit(X_train, y_train)

# Evaluate the model
y_pred = tree.predict(X_val)
accuracy = tree.accuracy(y_val.flatten(), y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Save the model
with open('decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(tree, model_file)

print("Model saved as 'decision_tree_model.pkl'.")
