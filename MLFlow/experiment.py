from flow import x_test_scaled, x_train_scaled, y_test, y_train
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import matplotlib.pyplot as plt

mlflow.set_experiment("Hyperparameters")
param_grid = {
    'n_estimators': [100, 150, 220],     # Number of trees
    'max_depth': [10, 20, 25],     # Maximum depth of the tree
    'min_samples_split': [10, 15, 20],     # Minimum number of samples required to split a node
    'min_samples_leaf': [5, 10, 14],       # Minimum number of samples at a leaf node
    'max_features': [.5, .7, .8],    # Number of features to consider for the best split
    # 'bootstrap': [True, False]           # Whether to use bootstrap samples
}

for i in range(len(param_grid["n_estimators"])):
  with mlflow.start_run():

    mlflow.log_param("n_estimators", param_grid["n_estimators"][i])
    mlflow.log_param("max_depth", param_grid["max_depth"][i])
    mlflow.log_param("min_samples_split", param_grid["min_samples_split"][i])
    mlflow.log_param("min_samples_leaf", param_grid["min_samples_leaf"][i])
    mlflow.log_param("max_features", param_grid["max_features"][i])

    rf = RandomForestRegressor(random_state = 42, oob_score = True,
                    bootstrap = True, n_jobs= -1,
                    n_estimators = param_grid["n_estimators"][i],
                    max_depth = param_grid["max_depth"][i],
                    min_samples_split = param_grid["min_samples_split"][i],
                    min_samples_leaf = param_grid["min_samples_leaf"][i],
                    max_features = param_grid["max_features"][i])

    rf.fit(x_train_scaled, y_train)

    y_train_pred = rf.predict(x_train_scaled)
    y_test_pred = rf.predict(x_test_scaled)

    mlflow.set_tag("Run No", i+1)
    mlflow.set_tag("Algorithm", "Random Forest")
    mlflow.set_tag("Phase", "Tuning")
    mlflow.set_tag("Model", "Regressor")
    mlflow.set_tag("Dataset", "Car Price")
    mlflow.set_tag("Features", "All")
    mlflow.set_tag("Target", "Selling Price")
    mlflow.set_tag("Encoder", "Target Encoder")
    mlflow.set_tag("Scaler", "MinMax Scaler")
    mlflow.set_tag("Hyperparameters", param_grid)
    mlflow.set_tag("Train Accuracy", r2_score(y_train, y_train_pred))
    mlflow.set_tag("Test Accuracy", r2_score(y_test, y_test_pred))

    mlflow.log_metric("train_mse", mean_squared_error(y_train, y_train_pred))
    mlflow.log_metric("test_mse", mean_squared_error(y_test, y_test_pred))
    mlflow.log_metric("train_accuracy", r2_score(y_train, y_train_pred))
    mlflow.log_metric("test_accuracy", r2_score(y_test, y_test_pred))

    mlflow.sklearn.log_model(rf, "model")

    # cm = ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
    # fig = cm.plot().figure_
    # mlflow.log_figure(fig, "confusion_matrix.png")

    # cm2 = ConfusionMatrixDisplay.from_prediction(y_train, y_train_pred)
    # cm2.figure_.savefig("confusion_matrix_train.png")
    # mlflow.log_artifact("confusion_matrix_train.png")

    # mlflow.log_artifact("scaler.joblib")
    # mlflow.log_artifact("encoder.joblib")
    # mlflow.log_artifact("model.joblib")
# prompt: create diff plots to compare output by model above with actual y_train values & save all plots in mlflow.log_artifact()

    # import matplotlib.pyplot as plt
    # Create a scatter plot to visualize the difference between actual and predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_train_pred)
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title('Actual vs. Predicted Selling Prices (Training Data)')
    # plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)  # Add a diagonal line for reference
    plt.savefig("scatter_plot_train.png")
    mlflow.log_artifact("scatter_plot_train.png")
    # plt.show()


    # You can also calculate the residuals (difference between actual and predicted)
    residuals = y_train - y_train_pred

    # And plot the distribution of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30)
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals (Training Data)')
    plt.savefig("residuals_hist_train.png")
    mlflow.log_artifact("residuals_hist_train.png")
    # plt.show()

    # Create a line plot to compare actual and predicted values over time (if applicable)
    plt.figure(figsize=(10, 6))
    plt.plot(y_train.values, label='Actual')
    plt.plot(y_train_pred, label='Predicted')
    plt.xlabel('Data Point Index')
    plt.ylabel('Selling Price')
    plt.title('Actual vs. Predicted Selling Prices (Training Data)')
    plt.legend()
    plt.savefig("line_plot_train.png")
    mlflow.log_artifact("line_plot_train.png")
    # plt.show()










