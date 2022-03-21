import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import model_selection
from sklearn import inspection

def FeatureImportance(X, y, spatial):    
    # Model    
    group_kfold = model_selection.GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    spatial_kfold = group_kfold.split(X, y, spatial)  # Create a nested list of train and test indices for each fold
    train_indices, test_indices = [list(traintest) for traintest in zip(*spatial_kfold)]
    spatial_cv = [*zip(train_indices,test_indices)]
    
    reg = ensemble.HistGradientBoostingRegressor(loss='poisson', max_bins=100)
    cv_res = model_selection.cross_validate(reg, X, y, cv=spatial_cv, 
                                            scoring='r2',
                                            n_jobs=48,
                                            return_estimator=True)
    
    # Importance of features
    imp_df = np.empty((0,X.shape[1]))
    for idx, estimator in enumerate(cv_res['estimator']):
        imp_res = inspection.permutation_importance(estimator, X, y, n_repeats=5, random_state=0, n_jobs=48)
        imp_df = np.vstack([imp_df, imp_res.importances_mean])

    return([cv_res['test_score'].mean()] + np.mean(imp_df, axis=0).tolist())

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
            'tas', 'tasmin', 'tasmax']

X = metadata[features]
exps = pd.get_dummies(metadata.Expedition, prefix='exp')
features.extend(exps.columns.tolist())
print(exps.columns.tolist())
X = np.hstack((X, exps))
print(X.shape)
print(len(features))

imp_df = pd.DataFrame(columns = ['Species', 'R2'] + features)

count=0
for ASV in data.index:
    count +=1
    print(count, end='\r')
    y = metadata.patch.map(lambda x: data.loc[ASV, data.columns == x].mean(axis=0))
    
    imp = FeatureImportance(X, y, metadata['gl_name'].values)
    imp_df.loc[imp_df.shape[0]] = [ASV] + imp

imp_df.to_csv('../data/feature_importance.csv',index=False)
