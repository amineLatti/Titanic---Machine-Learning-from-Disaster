import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

#Node Class
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
#Tree Class

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split['info_gain']>0:
                left_subtree = self.build_tree(best_split['left'], curr_depth+1)
                right_subtree = self.build_tree(best_split['right'], curr_depth+1)
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree, best_split['info_gain'])
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float('inf')
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                left = np.array([dataset[i] for i in range(num_samples) if feature_values[i]<=threshold])
                right = np.array([dataset[i] for i in range(num_samples) if feature_values[i]>threshold])
                if len(left)>0 and len(right)>0:
                    y, left_y, right_y = dataset[:,-1], left[:,-1], right[:,-1]
                    info_gain = self.information_gain(y, left_y, right_y)
                    if info_gain>max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['left'] = left
                        best_split['right'] = right
                        best_split['info_gain'] = info_gain
                        max_info_gain = info_gain
        return best_split

    def information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child)/len(parent)
        weight_r = len(r_child)/len(parent)
        return self.entropy(parent) - (weight_l*self.entropy(l_child)) - (weight_r*self.entropy(r_child))
    
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y==cls])/len(y)
            entropy += -p_cls*np.log2(p_cls)
        return entropy
    
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
    def predict(self, X):
        return [self.traverse_tree(x, self.root) for x in X]
    
    def traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index]<=node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
    
    def accuracy(self, y_true, y_pred):    
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

train_data = pd.read_csv("train.csv")

X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
y = train_data['Survived'].values.reshape(-1, 1)

X = np.nan_to_num(X)  # Replace NaN with zero (or could use other defaults)

# Encode categorical columns temporarily
# 1. Encode 'Sex': male=0, female=1
X[:, 1] = [0 if val == 'male' else 1 for val in X[:, 1]]

# 2. Encode 'Embarked': C=0, Q=1, S=2 (replace NaN with 2 as S is most frequent)
embarked_map = {'C': 0, 'Q': 1, 'S': 2}
X[:, 6] = [embarked_map.get(val, 2) for val in X[:, 6]]

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree
tree = DecisionTree(min_samples_split=5, max_depth=6)
tree.fit(X_train, y_train)

# Predict and evaluate on the validation set
y_pred = tree.predict(X_val)
accuracy = tree.accuracy(y_val.flatten(), y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Save the model
with open('decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(tree, model_file)

print("Model saved as 'decision_tree_model.pkl'.")