from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def split_data(reduced_data, target_col, test_size=0.2, random_state=42):
    X = reduced_data.drop(columns=[target_col])
    y = reduced_data[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, model_type='logistic_regression'):

    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    else:
        raise ValueError("Invalid model type. Choose from 'logistic_regression' or 'random_forest'.")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
   
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

