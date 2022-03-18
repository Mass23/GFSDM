import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import model_selection

def GetImportance(X, y, spatial):
    group_kfold = model_selection.GroupShuffleSplit(n_splits=10, random_state=0)
    spatial_kfold = group_kfold.split(X, y, spatial)  # Create a nested list of train and test indices for each fold
    train_indices, test_indices = [list(traintest) for traintest in zip(*spatial_kfold)]
    spatial_cv = [*zip(train_indices,test_indices)]
        
    reg = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500)
    pipe = Pipeline([('scaler', StandardScaler()), ('reg', reg)])

    cv_res = model_selection.cross_validate(pipe, X, y, cv=spatial_cv, 
                                            scoring=['r2','neg_mean_absolute_error','neg_mean_squared_error'],
                                            return_estimator=True, n_jobs=8)

    imp_df = np.empty((0,X.shape[1]))
    for idx, estimator in enumerate(cv_res['estimator']):
        imp_res = permutation_importance(estimator, X, y, n_repeats=1, random_state=0, n_jobs=8)
        imp_df = np.vstack([imp_df, imp_res.importances_mean])
    imp_means = imp_df.mean(axis=0)
    return(imp_means)

def RemoveCorrelations(X, importances):
    importances = importances[importances['importance'] > 0]
    features = list(importances['feature'].tolist())
    corrs = X[features].corr()
    
    for feature in features:
        feats_to_remove = []
        other_features = [feat for feat in features if feat != feature]
        for other_feature in other_features:
            if np.abs(corrs.loc[feature, other_feature]) >= 0.7:
                feats_to_remove.append(other_feature)
        features = [feat for feat in features if feat not in feats_to_remove]
    return(features)

def FeaturePrediction(X, y, spatial, feature_names, parameters, n_cores):    
    # Model    
    to_remove = np.isnan(y)
    X = X[~to_remove]
    y = y[~to_remove]
    spatial = spatial[~to_remove]

    print('Computing features importance...')
    imp_means = GetImportance(X, y, spatial)
    importances = np.array([(feature_names[i], value) for i, value in enumerate(imp_means)])
    
    importances = np.array([(feature, imp_means[i]) for i, feature in enumerate(features)], dtype = [('feature', '<U15'), ('importance', np.float64)])
    importances = np.sort(importances, order=['importance'])[::-1]
    print(importances)

    print('Removing correlated features with low predictive power...')
    final_features = RemoveCorrelations(X, importances)
    print(final_features)
    final_X = X[final_features]
    
    group_kfold = model_selection.GroupShuffleSplit(n_splits=10, random_state=0)
    spatial_kfold = group_kfold.split(X, y, spatial)  # Create a nested list of train and test indices for each fold
    train_indices, test_indices = [list(traintest) for traintest in zip(*spatial_kfold)]
    spatial_cv = [*zip(train_indices,test_indices)]
        
    reg = HistGradientBoostingRegressor()
    pipe = Pipeline([('scaler', StandardScaler()), ('reg', reg)])

    print('Searching best parameters in grid search ...')
    gs_cv = GridSearchCV(pipe, cv=spatial_cv, param_grid=parameters, n_jobs=n_cores,error_score='raise')
    gs_cv.fit(X,y)

    print(gs_cv.best_estimator_)

    print('Computing r2 ...')
    cv_res = model_selection.cross_validate(pipe, X, y, cv=spatial_cv, n_jobs=n_cores,
                                            scoring=['r2','neg_mean_absolute_error','neg_mean_squared_error'],
                                            return_estimator=True)
    print(cv_res['test_r2'].mean())
    print(cv_res['test_neg_mean_absolute_error'].mean())
    print(cv_res['test_neg_mean_squared_error'].mean())

# Data loading
metadata = pd.read_csv('../data/139GL_meta_clim.csv')
metadata['lat_sp [DD]'] = np.abs(metadata['lat_sp [DD]'])

features = ['lat_sp [DD]', 'ele_sp [m]', 'gl_sa [km2]', 'gl_cov [%]','bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19', 'bio1',
             'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8', 'bio9', 'fcf', 'fgd', 'scd', 'swe', 'pr', 
             'tas', 'tasmin', 'tasmax']
X = metadata[features]
spatial = metadata['gl_name'].values

parameters = {'reg__learning_rate': np.logspace(0.0001,0.5,10),
              'reg__l2_regularization': np.logspace(0.0001,10,20),
              'reg__max_iter': np.linspace(100,2000,10).astype(int),
              'reg__max_bins': [50,100,150,200]}

y = metadata['water_temp [°C]']
FeaturePrediction(X, y, spatial, features, parameters, 7)
