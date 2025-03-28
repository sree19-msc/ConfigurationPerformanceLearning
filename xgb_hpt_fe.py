import os
import time
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

# Import the pipeline from your separate script
from feature_engineering import feature_engineering_pipeline

def main():
    # Folders/systems
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    # Number of repeated splits
    num_repeats = 33
    # Fraction of data for training
    train_frac = 0.7
    # Seed for reproducibility
    random_seed = 1

    for current_system in systems:
        datasets_location = f'D:/ISE_lab2/lab2/datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f"\n> System: {current_system}, Dataset: {csv_file}, "
                  f"Training data fraction: {train_frac}, Number of repeats: {num_repeats}")

            # Load dataset
            data_path = os.path.join(datasets_location, csv_file)
            data = pd.read_csv(data_path)

            start_time = time.time()

            # Apply feature engineering; skip dataset if error
            try:
                data = feature_engineering_pipeline(data, target_column='time')
            except Exception as e:
                print(f"Skipping {csv_file} due to error in feature engineering: {e}")
                continue

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                # Random train/test split
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                # Separate features/target
                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Define hyperparameter grid
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }

                # Grid search
                grid_search = GridSearchCV(
                    estimator=XGBRegressor(),
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=3,
                    n_jobs=-1
                )

                # Fit best model
                grid_search.fit(training_X, training_Y)
                best_model = grid_search.best_estimator_

                # Predict and evaluate
                predictions = best_model.predict(testing_X)

                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            end_time = time.time()  # End timing
            total_time = end_time - start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)

            # Print final average metrics
            print('Average MAPE: {:.4f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE:  {:.4f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.4f}".format(np.mean(metrics['RMSE'])))
            print(f"Time taken: {minutes} min {seconds} sec")
            print('-' * 60)

if __name__ == "__main__":
    main()
