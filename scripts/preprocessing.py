import pandas as pd
import numpy as np

def preprocess_data(df, is_train=True):
    # Fill missing values
    df['Age'] = df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # Create new features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
    df['Cabin_Exists'] = np.where(df['Cabin'].isnull(), 0, 1)
    
    # Extract title from name
    df['Title'] = df['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
    rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Lady', 'Sir', 'Countess', 'Capt', 'Don', 'Jonkheer']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
    df['Title'] = df['Title'].fillna(0)
    
    # Drop irrelevant columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
    
    # Scale Fare
    df['Fare'] = np.log1p(df['Fare'])
    
    # If it's training data, separate features and target
    if is_train:
        X = df.drop(columns=['Survived'], errors='ignore')
        y = df['Survived']
        return X, y
    
    return df
