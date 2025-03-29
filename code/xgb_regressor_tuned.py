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
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33
    train_frac = 0.7
    random_seed = 1

    results = []

    for current_system in systems:
        datasets_location = '../datasets/{}'.format(current_system)
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f"\n> System: {current_system}, Dataset: {csv_file}, "
                  f"Training data fraction: {train_frac}, Number of repeats: {num_repeats}")

            data_path = os.path.join(datasets_location, csv_file)
            data = pd.read_csv(data_path)

            start_time = time.time()

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

                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }

                grid_search = GridSearchCV(
                    estimator=XGBRegressor(),
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=3,
                    n_jobs=-1
                )

                grid_search.fit(training_X, training_Y)
                best_model = grid_search.best_estimator_
                predictions = best_model.predict(testing_X)

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

            avg_mape = np.mean(metrics['MAPE'])
            avg_mae = np.mean(metrics['MAE'])
            avg_rmse = np.mean(metrics['RMSE'])

            # Print all metrics
            print("XGBoost Regressor Tuned Performance:")
            print('Average MAPE: {:.4f}'.format(avg_mape))
            print("Average MAE:  {:.4f}".format(avg_mae))
            print("Average RMSE: {:.4f}".format(avg_rmse))
            print(f"Time taken: {minutes} min {seconds} sec")
            print('-' * 60)

            # Store results for CSV
            results.append({
                'System': current_system,
                'Dataset': csv_file,
                'MAPE': round(avg_mape, 4),
                'MAE': round(avg_mae, 4),
                'RMSE': round(avg_rmse, 4)
            })

    # Save to output/xyz.csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'xgboost_tuned.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"\nAll results have been saved to {output_path}")

if __name__ == "__main__":
    main()

