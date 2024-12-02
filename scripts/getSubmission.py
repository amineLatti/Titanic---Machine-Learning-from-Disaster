from decision_tree import DecisionTree
import pickle
import pandas as pd
import numpy as np

# Load the test data
test_data = pd.read_csv("test.csv")

# Preprocessing the test data (same as the training data preprocessing)
def preprocess_test_data(df):
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')  # Replace missing embarked with 'S'
    
    # Encode 'Sex': male=0, female=1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Encode 'Embarked': C=0, Q=1, S=2
    embarked_map = {'C': 0, 'Q': 1, 'S': 2}
    df['Embarked'] = df['Embarked'].map(embarked_map)
    
    # Convert to numpy array
    return df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values

X_test = preprocess_test_data(test_data)

# Load the saved Decision Tree model
with open('decision_tree_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Predict survival on the test data
predictions = model.predict(X_test)

# Prepare the submission file
submission = test_data[['PassengerId']].copy()
submission['Survived'] = predictions

# Save the submission file
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
