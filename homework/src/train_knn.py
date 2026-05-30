#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

# importacion de librerias
from sklearn.neighbors import KNeighborsRegressor

from ._internals.calculate_metrics import calculate_metrics
from ._internals.prepare_data import prepare_data
from ._internals.print_metrics import print_metrics
from ._internals.save_model_if_better import save_model_if_better

x_train, x_test, y_train, y_test = prepare_data()

# entrenar el modelo
estimator = KNeighborsRegressor(n_neighbors=5)
estimator.fit(x_train, y_train)
save_model_if_better(estimator, x_test, y_test)

print()
print(estimator, ":", sep="")

# Metricas de error durante entrenamiento
mse, mae, r2 = calculate_metrics(estimator, x_train, y_train)
title = "Metricas de entrenamiento:"
print_metrics(mse, mae, r2, title)

# Metricas de error durante testing
mse, mae, r2 = calculate_metrics(estimator, x_test, y_test)
title = "Metricas de testing:"
print_metrics(mse, mae, r2, title)
