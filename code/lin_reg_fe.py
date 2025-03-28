import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from feature_engineering import feature_engineering_pipeline  # Import the feature engineering

def main():
    """
    Parameters:
    systems (list): List of systems containing CSV datasets.
    num_repeats (int): Number of times to repeat the evaluation for avoiding stochastic bias.
    train_frac (float): Fraction of data to use for training.
    random_seed (int): Initial random seed to ensure the results are reproducible
    """

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

            # Apply feature engineering
            data = feature_engineering_pipeline(data, target_column=data.columns[-1])

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                model = LinearRegression()
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

if __name__ == "__main__":
    main()
