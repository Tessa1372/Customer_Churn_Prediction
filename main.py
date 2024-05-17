# main.py

from modules.data_prep import load_data, handle_missing_values, encode_categorical_variables, scale_features
from modules.eda import calculate_churn_rate, visualize_churn_distribution, analyze_correlations
from modules.feature_eng import create_new_features, select_features
from modules.modeling import split_data, train_model, evaluate_model
from modules.deployment import save_model

def main():
    # Step 1: Data Acquisition and Pre-processing
    data = load_data("data/telecom_churn.csv")
    data = handle_missing_values(data)
    data = encode_categorical_variables(data, categorical_cols=['Churn','ContractRenewal', 'DataPlan'])
    data = scale_features(data, numeric_cols=['AccountWeeks', 'DataUsage', 'CustServCalls', 'DayMins', 'DayCalls', 'MonthlyCharge', 'OverageFee', 'RoamMins'])
    
    # Step 2: Exploratory Data Analysis
    churn_rate = calculate_churn_rate(data)
    visualize_churn_distribution(data, 'ContractRenewal')
    analyze_correlations(data)
    
    # Step 3: Feature Engineering
    data = create_new_features(data)
    selected_features = select_features(data, feature_list=['Churn','AccountWeeks', 'DataUsage', 'CustServCalls', 'DayMins', 'DayCalls', 'MonthlyCharge', 'OverageFee', 'RoamMins', 'TotalMinutes', 'ContractRenewal', 'DataPlan'])
    
    # Step 4: Model Building and Evaluation
    X_train, X_test, y_train, y_test = split_data(selected_features, target_col='Churn')
    model = train_model(X_train, y_train, model_type='logistic_regression')
    evaluate_model(model, X_test, y_test)
    
    # Step 5: Model Deployment and Monitoring
    save_model(model, 'churn_prediction_model.pkl')

if __name__ == "__main__":
    main()
