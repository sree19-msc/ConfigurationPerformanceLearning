import pandas as pd
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np

def main():
    # Parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33
    train_frac = 0.7
    random_seed = 1

    for current_system in systems:
        datasets_location = f'D:/ISE_lab2/lab2/datasets/{current_system}'

        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f"\n> System: {current_system}, Dataset: {csv_file}, Training data fraction: {train_frac}, Number of repeats: {num_repeats}")

            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Normal XGBoost model (no hyperparameter tuning)
                model = XGBRegressor()
                model.fit(training_X, training_Y)

                predictions = model.predict(testing_X)

                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            print('Average MAPE: {:.4f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE: {:.4f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.4f}".format(np.mean(metrics['RMSE'])))
            print('-' * 60)

if __name__ == "__main__":
    main()
