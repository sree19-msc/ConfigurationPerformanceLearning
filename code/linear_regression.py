import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np

def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33
    train_frac = 0.7
    random_seed = 1

    # List to store final results
    results = []

    for current_system in systems:
        datasets_location = '../datasets/{}'.format(current_system)
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')] 

        for csv_file in csv_files:
            print('\n> System: {}, Dataset: {}, Training data fraction: {}, Number of repeats: {}'.format(current_system, csv_file, train_frac, num_repeats))
            data = pd.read_csv(os.path.join(datasets_location, csv_file))
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

            avg_mape = np.mean(metrics['MAPE'])
            avg_mae = np.mean(metrics['MAE'])
            avg_rmse = np.mean(metrics['RMSE'])

            # Print metrics with 4 decimal places
            print("Linear Regression Performance:")
            print('Average MAPE: {:.4f}'.format(avg_mape))
            print('Average MAE: {:.4f}'.format(avg_mae))
            print('Average RMSE: {:.4f}'.format(avg_rmse))

            # Append the result for this dataset to the results list
            results.append({
                'System': current_system,
                'Dataset': csv_file,
                'MAPE': round(avg_mape, 4),
                'MAE': round(avg_mae, 4),
                'RMSE': round(avg_rmse, 4)
            })

    # Ensure 'output' folder exists in the same location as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Save the CSV to the 'output' folder
    output_path = os.path.join(output_dir, 'linear_regression.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"\nAll results have been saved to {output_path}")

if __name__ == "__main__":
    main()
