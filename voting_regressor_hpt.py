import pandas as pd
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

# Define hyperparameter grids for XGBoost and Random Forest
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5, 10]
}

# Function to tune individual models
def tune_model(model, param_grid, training_X, training_Y):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,
        n_jobs=4  # Use 4 CPU cores to speed up execution
    )
    grid_search.fit(training_X, training_Y)
    return grid_search.best_params_

def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33
    train_frac = 0.7
    random_seed = 1 

    for current_system in systems:
        datasets_location = 'D:/ISE_lab2/lab2/datasets/{}'.format(current_system)

        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print('\n> System: {}, Dataset: {}, Training data fraction: {}, Number of repeats: {}'.format(
                current_system, csv_file, train_frac, num_repeats))

            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed*current_repeat)
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Tune XGBoost and Random Forest
                best_xgb_params = tune_model(XGBRegressor(), xgb_param_grid, training_X, training_Y)
                best_rf_params = tune_model(RandomForestRegressor(), rf_param_grid, training_X, training_Y)

                # Create models with the best parameters
                model_xgb = XGBRegressor(**best_xgb_params)
                model_rf = RandomForestRegressor(**best_rf_params)
                model_lr = LinearRegression()  # No hyperparameters to tune for Linear Regression

                # Define the Voting Regressor
                voting_reg = VotingRegressor(
                    estimators=[('lr', model_lr), ('xgb', model_xgb), ('rf', model_rf)],
                    weights=[1, 2, 2]  # Modify weights to optimize performance
                )

                # Train the Voting Regressor
                voting_reg.fit(training_X, training_Y)

                predictions = voting_reg.predict(testing_X)

                # Compute Metrics
                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Print Average Metrics
            print('Average MAPE: {:.4f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE: {:.4f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.4f}".format(np.mean(metrics['RMSE'])))

if __name__ == "__main__":
    main()
