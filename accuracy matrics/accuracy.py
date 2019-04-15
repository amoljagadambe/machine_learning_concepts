from sklearn.metrics import mean_absolute_error

#MAE ( mean absolute error)
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]

result = mean_absolute_error(y_true, y_pred)
print(result)

mutipoint = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
print(mutipoint)

#RMSE (root mean squrred error)