import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
from utils import *

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
best_X_sel = None
best_sel_features = None

modelos = [
    ('LinearSVR', LinearSVR(random_state=0)),
    ('GaussianProcessRegressor', GaussianProcessRegressor()),
    ('KNeighborsRegressor', KNeighborsRegressor()),
    ('SVR', SVR()),
    ('RandomForestRegressor', RandomForestRegressor(random_state=0)),
    ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=0)),
    ('LinearRegression', LinearRegression()),
    ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=0))
]
scores = []
#%%
for name, model in modelos:
    print(f"Evaluating {name}...")
    best_n_features, best_X_sel, best_sel_features = bestNFeatures(model, X_train_sc, y_train)
    score = cross_val_score(model, best_X_sel, y_train, cv=10, scoring='r2').mean()
    scores.append((name, model, score, best_X_sel, best_sel_features))

print("Fim das avaliações")

#%%
scores.sort(key=lambda x: x[2], reverse=True)

for name, _, score, _, sel_features in scores:
    print(f"{name}: R² = {score:.4f}, Selected features: {list(sel_features)}")
#%% modelo Final
scores_finais = []
for name, model, score, best_X_sel, best_sel_features in scores:
    if(score > 0.80):
        modelo_final = model
        modelo_final.fit(best_X_sel, y_train)

        y_pred = modelo_final.predict(X_test_sc[best_sel_features])
        r2 = modelo_final.score(X_test_sc[best_sel_features], y_test)
        rmse = (mean_squared_error(y_test, y_pred) ** 0.5)
        mae = mean_absolute_error(y_test, y_pred)
        scores_finais.append((name, score, r2, rmse, mae))
        print(f"Avaliando performace {name}...")
        print('r2', r2)
        print('rmse', rmse)
        print('mae', mae)
#%%
df_results = pd.DataFrame(scores_finais, columns=["Model", "Train_R2", "Test_R2", "RMSE", "MAE"])
df_results.sort_values(by="Test_R2", ascending=False)
print(df_results)