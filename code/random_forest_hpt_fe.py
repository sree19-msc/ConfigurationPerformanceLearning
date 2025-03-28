import os
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from feature_engineering import feature_engineering_pipeline  # Import your pipeline

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Function to tune Random Forest using GridSearchCV
def tune_random_forest(training_X, training_Y):
    model = RandomForestRegressor()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',  # Optimize for MAE
        cv=3,  # 3-fold cross-validation
        n_jobs=-1  # Use all available processors
    )
    
    grid_search.fit(training_X, training_Y)
    return grid_search.best_params_  # Return the best parameters

def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33
    train_frac = 0.7
    random_seed = 1

    for current_system in systems:
        datasets_location = f'D:/ISE_lab2/lab2/datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f"\n> System: {current_system}, Dataset: {csv_file}, Training data fraction: {train_frac}, Number of repeats: {num_repeats}")
            start_time = time.time()

            # Load dataset
            data_path = os.path.join(datasets_location, csv_file)
            data = pd.read_csv(data_path)

            # Universal approach: rename 'throughput' -> 'time' if 'time' not in columns
            if 'time' not in data.columns and 'throughput' in data.columns:
                data.rename(columns={'throughput': 'time'}, inplace=True)

            # Apply feature engineering, skip if any error
            try:
                data = feature_engineering_pipeline(data, target_column='time')
            except Exception as e:
                print(f"Skipping {csv_file} due to error in feature engineering: {e}")
                continue

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            # Repeat the process num_repeats times
            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Hyperparameter tuning for Random Forest
                best_params = tune_random_forest(training_X, training_Y)

                # Train with the best hyperparameters
                model = RandomForestRegressor(**best_params)
                model.fit(training_X, training_Y)

                predictions = model.predict(testing_X)

                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            end_time = time.time()
            total_time = end_time - start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)

            print(f"Average MAPE: {np.mean(metrics['MAPE']):.4f}")
            print(f"Average MAE:  {np.mean(metrics['MAE']):.4f}")
            print(f"Average RMSE: {np.mean(metrics['RMSE']):.4f}")
            print(f"Time taken:   {minutes} min {seconds} sec")
            print("-" * 60)

if __name__ == "__main__":
    main()
