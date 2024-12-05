import pandas as pd
import pickle
from preprocessing import preprocess_data

# Load test data
test_data = pd.read_csv('test.csv')

# Preprocess test data
X_test = preprocess_data(test_data, is_train=False)

# Load the saved model
with open('decision_tree_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Predict survival on the test data
predictions = model.predict(X_test.values)

# Prepare submission file
submission = test_data[['PassengerId']].copy()
submission['Survived'] = predictions

# Save submission file
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
