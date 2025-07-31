import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE

#%% Separação dos dados Teste e treinamento
df = pd.read_csv('data/dataset_problema2.csv')

X = df.drop(['id', 'Severidade'], axis=1)
y = df['Severidade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% padronização dos dados
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

X_train_sc = pd.DataFrame(X_train_sc)
X_train_sc.columns = X_train.columns

X_test_sc = pd.DataFrame(X_test_sc)
X_test_sc.columns = X_test.columns


#%% seleção de features
best_n_features = 0
best_score = 0
best_X_sel = None
max_n_features = len(X_train_sc.columns)+1

for n_features in range(1, max_n_features):

    modelo_linear = LinearRegression()

    selector = RFE(modelo_linear, n_features_to_select=n_features, step=1)

    selector = selector.fit(X_train_sc, y_train)

    mask = selector.support_

    features = X_train_sc.columns

    selected_features = features[mask]

    X_sel = X_train_sc[selected_features]

    score = cross_val_score(modelo_linear, X_sel, y_train, cv=10, scoring='r2')

    print(np.mean(score))
    print(n_features)
    if(np.mean(score) > best_score):
        best_score = np.mean(score)
        best_n_features = n_features
        best_X_sel = X_sel

print(best_n_features)
print(best_score)
print(best_X_sel)

#%% validação cruzada
model_linear = LinearRegression()

score = cross_val_score(model_linear, best_X_sel, y_train, cv=10, scoring='r2')

print(score)
print(np.mean(score))

#%% modelo Final
model_linear = LinearRegression()
model_linear.fit(best_X_sel, y_train)

#%% Validação do modelo
y_pred = model_linear.predict(X_test_sc[best_X_sel])

r2 = model_linear.score(X_test_sc[best_X_sel], y_test)

rmse = (mean_squared_error(y_test, y_pred))**0.5

mae = mean_absolute_error(y_test, y_pred)**0.5

print(r2)

print(rmse)
print(mae)


