import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [5, 10, 20],  # Tree depth
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Function to tune Random Forest using GridSearchCV
def tune_random_forest(training_X, training_Y):
    model = RandomForestRegressor()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',  # Optimize for MAE
        cv=3,  # 3-fold Cross-validation
        n_jobs=-1  # Use all available processors
    )
    
    grid_search.fit(training_X, training_Y)
    return grid_search.best_params_  # Return the best parameters


def main():
    """
    Parameters:
    systems (list): List of systems containing CSV datasets.
    num_repeats (int): Number of times to repeat the evaluation for avoiding stochastic bias.
    train_frac (float): Fraction of data to use for training.
    random_seed (int): Initial random seed to ensure the results are reproducible.
    """

    # Specify the parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 33  # Using 33 repetitions as per the requirement
    train_frac = 0.7  # 70% of data for training
    random_seed = 1  # Altered for each repeat

    for current_system in systems:
        datasets_location = 'D:/ISE_lab2/lab2/datasets/{}'.format(current_system)  # Location of datasets

        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]  # List all CSV files

        for csv_file in csv_files:
            print('\n> System: {}, Dataset: {}, Training data fraction: {}, Number of repeats: {}'.format(
                current_system, csv_file, train_frac, num_repeats))

            data = pd.read_csv(os.path.join(datasets_location, csv_file))  # Load dataset

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}  # Store results for repeated evaluations

            for current_repeat in range(num_repeats):  # Repeat n times
                # Split data into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                # Split features (X) and target (Y)
                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # **Hyperparameter tuning for Random Forest**
                best_params = tune_random_forest(training_X, training_Y)

                # Train Random Forest with the best hyperparameters
                model = RandomForestRegressor(**best_params)

                model.fit(training_X, training_Y)  # Train the model

                predictions = model.predict(testing_X)  # Predict

                # Calculate evaluation metrics
                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                # Store the metrics
                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Calculate the average of the metrics for all repeats
            print('Average MAPE: {:.4f}'.format(np.mean(metrics['MAPE'])))
            print("Average MAE: {:.4f}".format(np.mean(metrics['MAE'])))
            print("Average RMSE: {:.4f}".format(np.mean(metrics['RMSE'])))

if __name__ == "__main__":
    main()
