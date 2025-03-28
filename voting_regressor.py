import pandas as pd
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

def main():
    """
    Trains a Voting Regressor combining Linear Regression, XGBoost, and Random Forest,
    then evaluates its performance on different datasets.
    """

    # Define systems & parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33  # Number of times to repeat for avoiding stochastic bias
    train_frac = 0.7  # Fraction of data to use for training

    for current_system in systems:
        datasets_location = f'D:/ISE_lab2/lab2/datasets/{current_system}'  # Adjust dataset path
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]  # Get CSV files

        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}, Training data fraction: {train_frac}, Number of repeats: {num_repeats}')

            data = pd.read_csv(os.path.join(datasets_location, csv_file))  # Load dataset

            # Initialize a dictionary to store results
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for _ in range(num_repeats):  # Repeat multiple times
                # Train-test split
                train_data = data.sample(frac=train_frac, random_state=_)
                test_data = data.drop(train_data.index)

                # Separate features (X) and target (Y)
                training_X, testing_X = train_data.iloc[:, :-1], test_data.iloc[:, :-1]
                training_Y, testing_Y = train_data.iloc[:, -1], test_data.iloc[:, -1]

                # Define models
                lin_model = LinearRegression()
                xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
                rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

                # Create Voting Regressor
                voting_model = VotingRegressor(estimators=[
                    ('Linear Regression', lin_model),
                    ('XGBoost', xgb_model),
                    ('Random Forest', rf_model)
                ])

                # Train Voting Regressor
                voting_model.fit(training_X, training_Y)

                # Make predictions
                voting_pred = voting_model.predict(testing_X)

                # Compute metrics
                metrics['MAPE'].append(mean_absolute_percentage_error(testing_Y, voting_pred))
                metrics['MAE'].append(mean_absolute_error(testing_Y, voting_pred))
                metrics['RMSE'].append(np.sqrt(mean_squared_error(testing_Y, voting_pred)))

            # Print final average metrics for the combined VotingRegressor model
            print("Voting Regressor Performance:")
            print(f"  - Average MAPE: {np.mean(metrics['MAPE']):.4f}")
            print(f"  - Average MAE: {np.mean(metrics['MAE']):.4f}")
            print(f"  - Average RMSE: {np.mean(metrics['RMSE']):.4f}")
            print("-" * 60)

if __name__ == "__main__":
    main()
