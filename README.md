# prediccionesSalud

Predicciones en una base de datos utilizando diversos modelos de Scikit-Learn y Pandas.

# Problema

(tomado de https://github.com/soyHenry/Datathon)

Un importante Centro de Salud lo ha contratado con el fin de poder predecir si un paciente tendrá una estancia hospitalaria prolongada o no, utilizando la información contenida en el dataset asociado, la cual recaba una muestra histórica de sus pacientes, para poder administrar la demanda de camas en el hospital según la condición de los pacientes recientemente ingresados.

Para esto, se define que un paciente posee estancia hospitalaria prolongada si ha estado hospitalizado más de 8 días. Por lo que debe generar dicha variable categórica y luego categorizar los pacientes según las variables que usted considere necesarias, justificando dicha elección.​

# Estructura del proyecto

En la carpeta medical_predictions se encuentran dos scripts de Python, generate_models.py y generate_peraquets.py, los cuales generan las predicciones y los datos (en formato Parquet) con los cuales se entrena el modelo, respectivamente.

La carpeta notebooks contiene el notebook en el cual se realizo el EDA (initial_data_exploration.ipynb), y el notebook que se utilizo para comparar y contrastar los distintos modelos de clasificación disponibles.

Finalmente, la carpeta data contiene los datos en bruto, los datos generados en formato parquet, y las predicciones finales realizadas sobre los datos de prueba.