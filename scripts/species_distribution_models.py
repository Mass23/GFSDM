import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import model_selection

def GetImportance(X, y, spatial):
    group_kfold = model_selection.GroupShuffleSplit(n_splits=5, test_size = 0.25, random_state=0)
    spatial_kfold = group_kfold.split(X, y, spatial)  # Create a nested list of train and test indices for each fold
    train_indices, test_indices = [list(traintest) for traintest in zip(*spatial_kfold)]
    spatial_cv = [*zip(train_indices,test_indices)]
        
    reg = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500)
    pipe = Pipeline([('scaler', StandardScaler()), ('reg', reg)])

    cv_res = model_selection.cross_validate(pipe, X, y, cv=spatial_cv, 
                                            scoring=['r2','neg_mean_absolute_error','neg_mean_squared_error'],
                                            return_estimator=True, n_jobs=48)

    imp_df = np.empty((0,X.shape[1]))
    for idx, estimator in enumerate(cv_res['estimator']):
        imp_res = permutation_importance(estimator, X, y, n_repeats=1, random_state=0, n_jobs=48)
        imp_df = np.vstack([imp_df, imp_res.importances_mean])
    imp_means = imp_df.mean(axis=0)
    return(imp_means)

def RemoveCorrelations(X, importances):
    importances = importances[importances['importance'] > 0]
    features = list(importances['feature'].tolist())
    print(features)
    corrs = X[features].corr()
    
    feats_to_remove = []
    for feature in features:
        if feature not in feats_to_remove:
            other_features = [feat for feat in features if feat not in [feature] + feats_to_remove]
            for other_feature in other_features:
                if np.abs(corrs.loc[feature, other_feature]) >= 0.7:
                    feats_to_remove.append(other_feature)
    features = [feat for feat in features if feat not in feats_to_remove]
    return(features)

# Data loading
metadata = pd.read_csv('../data/139GL_meta_clim.csv')
metadata = metadata[~np.isnan(metadata['sba [cells g-1]'])]

data = pd.read_csv('../data/NOMIS_16S_table_0122_filtered.csv')
tax_data = pd.read_csv('../data/NOMIS_16S_taxonomy_0122_filtered.csv')

metadata = metadata[metadata.patch.isin(data.columns)]
data.index = data.Feature_ID
print(data.shape)

# Data formatting
features = ['water_temp [°C]', 'ph [pH]', 'do_sat [saturation]', 'w_co2 [mATM]', 'conductivity [uS cm -1]',
            'turb [NTU]', 'lat_sp [DD]', 'ele_sp [m]', 'gl_sa [km2]', 'gl_cov [%]', 'chla [ug g-1]', 
            'n3_srp [ug l-1]', 'n4_nh4 [ug l-1]', 'n5_no3 [ug l-1]', 'n6_no2 [ug l-1]', 
            'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19', 'bio1',
            'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8', 'bio9', 'fcf', 'fgd', 'scd', 'swe', 'pr', 
            'tas', 'tasmin', 'tasmax','gl_a [km2]','sn_sp_dist [m]']

X = metadata[features]
exps = pd.get_dummies(metadata.Expedition, prefix='exp')
features.extend(exps.columns.tolist())
print(exps.columns.tolist())
X = pd.concat([X, exps], axis=1)
print(X.head())
print(len(features))

spatial = metadata['gl_name'].values

parameters = {'reg__learning_rate': np.logspace(0.0001,0.5,10),
              'reg__l2_regularization': np.logspace(0.0001,10,10),
              'reg__max_iter': np.linspace(200,2000,10).astype(int),
              'reg__max_bins': [50,100,150,200]}

n_cores=7
final_models = pd.DataFrame(columns = ['ASV', 'r2', 'neg_abs_err', 'neg_squared_err', 'features', 'features_importance'])

for ASV in data.index:
    print(ASV)
    y = np.log(metadata.patch.map(lambda x: data.loc[ASV, data.columns == x].mean(axis=0)) + 1)

    print('Computing features importance...')
    imp_means = GetImportance(X, y, spatial)
    importances = np.array([(features[i], value) for i, value in enumerate(imp_means)])
    
    importances = np.array([(feature, imp_means[i]) for i, feature in enumerate(features)], dtype = [('feature', '<U25'), ('importance', np.float64)])
    importances = np.sort(importances, order=['importance'])[::-1]
    print(importances)

    print('Removing correlated features with low predictive power...')
    final_features = RemoveCorrelations(X, importances)
    print(final_features)
    final_X = X[final_features]
    
    group_kfold = model_selection.GroupShuffleSplit(n_splits=5, test_size = 0.25, random_state=0)
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


    imp_df = np.empty((0,X.shape[1]))
    for idx, estimator in enumerate(cv_res['estimator']):
        imp_res = permutation_importance(estimator, X, y, n_repeats=1, random_state=0, n_jobs=48)
        imp_df = np.vstack([imp_df, imp_res.importances_mean])
    final_importances = imp_df.mean(axis=0)

    final_models = pd.read_csv('../final_models.csv')
    final_models = pd.concat([final_models, pd.DataFrame({'ASV':ASV, 
                                                          'best_model':str(gs_cv.best_estimator_),
                                                          'r2':cv_res['test_r2'].mean(), 
                                                          'neg_abs_err':cv_res['test_neg_mean_absolute_error'].mean(), 
                                                          'neg_squared_err':cv_res['test_neg_mean_squared_error'].mean(), 
                                                          'features':str(final_features), 
                                                          'features_importance': str(final_importances)})], index=[ASV])  
    final_models.to_csv('../final_models.csv')
