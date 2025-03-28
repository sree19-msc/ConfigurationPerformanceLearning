import os
import time
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

# Import your feature engineering pipeline
from feature_engineering import feature_engineering_pipeline

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
        n_jobs=4  # or -1 to use all CPU cores
    )
    grid_search.fit(training_X, training_Y)
    return grid_search.best_params_

def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33
    train_frac = 0.7
    random_seed = 1

    for current_system in systems:
        datasets_location = f'D:/ISE_lab2/lab2/datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f"\n> System: {current_system}, Dataset: {csv_file}, "
                  f"Training data fraction: {train_frac}, Number of repeats: {num_repeats}")
            
            # Start timing
            start_time = time.time()

            data_path = os.path.join(datasets_location, csv_file)
            data = pd.read_csv(data_path)

            # If there's no 'time' column but 'throughput' is present, rename for consistency
            if 'time' not in data.columns and 'throughput' in data.columns:
                data.rename(columns={'throughput': 'time'}, inplace=True)

            # Apply feature engineering, skip if any error arises
            try:
                data = feature_engineering_pipeline(data, target_column='time')
            except Exception as e:
                print(f"Skipping {csv_file} due to error in feature engineering: {e}")
                continue

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Tune XGBoost and Random Forest
                best_xgb_params = tune_model(XGBRegressor(), xgb_param_grid, training_X, training_Y)
                best_rf_params  = tune_model(RandomForestRegressor(), rf_param_grid, training_X, training_Y)

                # Create models with the best hyperparameters
                model_xgb = XGBRegressor(**best_xgb_params)
                model_rf  = RandomForestRegressor(**best_rf_params)
                model_lr  = LinearRegression()  # No tuning for LinearRegression

                # Voting Regressor
                voting_reg = VotingRegressor(
                    estimators=[('lr', model_lr), ('xgb', model_xgb), ('rf', model_rf)],
                    weights=[1, 2, 2]  # Adjust weights as desired
                )

                voting_reg.fit(training_X, training_Y)
                predictions = voting_reg.predict(testing_X)

                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae  = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # End timing
            end_time = time.time()
            duration = end_time - start_time
            minutes = int(duration // 60)
            seconds = int(duration % 60)

            # Print final average metrics
            print(f"Average MAPE: {np.mean(metrics['MAPE']):.4f}")
            print(f"Average MAE:  {np.mean(metrics['MAE']):.4f}")
            print(f"Average RMSE: {np.mean(metrics['RMSE']):.4f}")
            print(f"Time taken:   {minutes} min {seconds} sec")
            print("-" * 60)

if __name__ == "__main__":
    main()
