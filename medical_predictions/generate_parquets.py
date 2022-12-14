import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Cargamos los datos

hsptl_test_df = pd.read_csv("../data/raw/hospitalizaciones_test.csv")
hsptl_train_df = pd.read_csv("../data/raw/hospitalizaciones_train.csv")

# Codificamos las categorias, según si son ordinales o nominales
# Empezando por las ordinales

enc = OrdinalEncoder()
hsptl_train_df.Age = enc.fit_transform(hsptl_train_df[["Age"]])
enc.set_params(categories=[['Minor','Moderate','Extreme']])
hsptl_train_df[['Severity of Illness']] = enc.fit_transform(hsptl_train_df[['Severity of Illness']])
enc.set_params(categories=[['No','Yes']])
hsptl_train_df.Insurance = enc.fit_transform(hsptl_train_df[['Insurance']])

# y siguiendo por las nominales

ohe = OneHotEncoder()

feature_array = ohe.fit_transform(hsptl_train_df[["gender"]]).toarray()
feature_labels = np.array(ohe.categories_).ravel()
feature_labels = np.delete(feature_labels, 2)
feature_labels = np.append(feature_labels, 'OtherGenres')
ohe_admission = pd.DataFrame(feature_array, columns=feature_labels)
hsptl_train_df = pd.concat([hsptl_train_df, ohe_admission], axis=1)
hsptl_train_df.drop('gender',axis=1, inplace=True)


feature_array = ohe.fit_transform(hsptl_train_df[["Type of Admission"]]).toarray()
feature_labels = np.array(ohe.categories_).ravel()
ohe_admission = pd.DataFrame(feature_array, columns=feature_labels)
hsptl_train_df = pd.concat([hsptl_train_df, ohe_admission], axis=1)
hsptl_train_df.drop('Type of Admission',axis=1, inplace=True)

feature_array = ohe.fit_transform(hsptl_train_df[["health_conditions"]]).toarray()
feature_labels = np.array(ohe.categories_).ravel()
ohe_admission = pd.DataFrame(feature_array, columns=feature_labels)
ohe_admission.drop(['None'], axis=1, inplace=True)
ohe_admission.rename(columns={'Other':'OtherDiseases'},inplace=True)
hsptl_train_df = pd.concat([hsptl_train_df, ohe_admission], axis=1)
hsptl_train_df.drop('health_conditions',axis=1, inplace=True)

# Hacemos el mismo proceso con los datos de prueba

enc = OrdinalEncoder()
hsptl_test_df.Age = enc.fit_transform(hsptl_test_df[["Age"]])
enc.set_params(categories=[['Minor','Moderate','Extreme']])
hsptl_test_df[['Severity of Illness']] = enc.fit_transform(hsptl_test_df[['Severity of Illness']])
enc.set_params(categories=[['No','Yes']])
hsptl_test_df.Insurance = enc.fit_transform(hsptl_test_df[['Insurance']])

feature_array = ohe.fit_transform(hsptl_test_df[["gender"]]).toarray()
feature_labels = np.array(ohe.categories_).ravel()
feature_labels = np.delete(feature_labels, 2)
feature_labels = np.append(feature_labels, 'OtherGenres')
ohe_admission = pd.DataFrame(feature_array, columns=feature_labels)
hsptl_test_df = pd.concat([hsptl_test_df, ohe_admission], axis=1)
hsptl_test_df.drop('gender',axis=1, inplace=True)

feature_array = ohe.fit_transform(hsptl_test_df[["Type of Admission"]]).toarray()
feature_labels = np.array(ohe.categories_).ravel()
ohe_admission = pd.DataFrame(feature_array, columns=feature_labels)
hsptl_test_df = pd.concat([hsptl_test_df, ohe_admission], axis=1)
hsptl_test_df.drop('Type of Admission',axis=1, inplace=True)

feature_array = ohe.fit_transform(hsptl_test_df[["health_conditions"]]).toarray()
feature_labels = np.array(ohe.categories_).ravel()
ohe_admission = pd.DataFrame(feature_array, columns=feature_labels)
ohe_admission.drop(['None'], axis=1, inplace=True)
ohe_admission.rename(columns={'Other':'OtherDiseases'},inplace=True)
hsptl_test_df = pd.concat([hsptl_test_df, ohe_admission], axis=1)
hsptl_test_df.drop('health_conditions',axis=1, inplace=True)

# Ya habiendo preparados los dataframes, los pasamos a un parquet para cargarlos después al generar nuestro modelo

hsptl_train_df.to_parquet('../data/processed/hsptl_train.parquet')
hsptl_test_df.to_parquet('../data/processed/hsptl_test.parquet')