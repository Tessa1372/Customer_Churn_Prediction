import joblib
import pandas as pd


def save_model(model, file_path):
    joblib.dump(model, file_path)

def load_model(file_path):
    return joblib.load(file_path)

def predict(input_data, model):
    # Implement model prediction logic here
    predictions = model.predict(input_data)
    return predictions

if __name__ == "__main__":
    # Sample input data including the newly added 'TotalMinutes' feature
    input_data = pd.DataFrame({
        'AccountWeeks': [70, 50, 90, 60, 110],
        'DataUsage': [2.5, 1.2, 3.0, 0.5, 2.8],
        'CustServCalls': [0, 2, 1, 3, 0],
        'DayMins': [200, 150, 220, 100, 250],
        'DayCalls': [100, 120, 80, 90, 110],
        'MonthlyCharge': [50, 40, 60, 30, 70],
        'OverageFee': [10, 5, 15, 3, 12],
        'RoamMins': [5, 2, 8, 0, 10],
        'TotalMinutes': [300, 200, 350, 150, 400],
        'ContractRenewal': [1, 0, 1, 1, 0],  # 1 represents renewed, 0 represents not renewed
        'DataPlan': [1, 0, 1, 0, 1]  # 1 represents subscribed, 0 represents not subscribed
    })

    # Load the model
    model_file_path = 'churn_prediction_model.pkl'  # Replace with the path to your model file
    loaded_model = load_model(model_file_path)

    # Make predictions
    predictions = predict(input_data, loaded_model)
    print(predictions)

import modelbit
mb = modelbit.login()
mb.deploy(predict)