import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pylab as pl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from itertools import compress
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge
import sys
from sklearn import tree
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import pearsonr
import time
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

start = time.time()
''' 
    Data import 
'''
path_trainData = sys.argv[1]
path_testData = sys.argv[2]
trainData = pd.read_csv(path_trainData)
testData = pd.read_csv(path_testData)

''' 
    trainData preprocessing 
'''


def trainProcess(data, targetCol, outlierFilterMethod='ordinal'):
    targetColSeries = data[targetCol]
    data = data.drop([targetCol], axis=1)
    # Missing value handling
    data = data.dropna(thresh=len(data) * 0.9, axis=1)  # Remove fields with high missing rate

    # Separate continuous and discrete data
    trainData_continuous = data.loc[:,
                           [dtype.name != 'object' for dtype in data.dtypes.values.tolist()]]  # continuous data
    trainData_discrete = data.loc[:, [dtype.name == 'object' for dtype in data.dtypes.values.tolist()]]  # discrete data
    if (trainData_continuous.shape[1] != 0):  # For missing values of continuous variables, fill in with the mean
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        trainData_fillNan_continuous = pd.DataFrame(imp.fit_transform(trainData_continuous),
                                                    columns=trainData_continuous.columns)
    else:
        trainData_fillNan_continuous = trainData_continuous
    if (trainData_discrete.shape[
        1] != 0):  # For missing values of discrete features, treat the missing values as a separate category
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='NA')
        trainData_fillNan_discrete = pd.DataFrame(imp.fit_transform(trainData_discrete),
                                                  columns=trainData_discrete.columns)
    else:
        trainData_fillNan_discrete = trainData_discrete
    data = pd.concat([trainData_fillNan_continuous, trainData_fillNan_discrete], axis=1)  # concat
    targetColSeries = targetColSeries.filter(items=list(data.index.values), axis=0)

    # Handling Typed Features
    def dfEncodeDeal(data, calType='oneHot'):
        if (calType == 'oneHot'):
            for item in data.dtypes.unique():
                if (item.type.__name__ == 'object_'):
                    categoryColList = data.select_dtypes(item.type.__name__).columns.to_list()
                    #                     print('####################Print discrete value ranges of categorical features：####################')
                    #                     for col in categoryColList:
                    #                         print(data[col].unique())
                    df_onehot = pd.get_dummies(data[categoryColList])
                    data.drop(categoryColList, axis=1, inplace=True)
                    data = pd.concat([data, df_onehot], axis=1)
                    return data
            return None
        elif (calType == 'ordinal'):
            for item in data.dtypes.unique():
                if (item.type.__name__ == 'object_'):
                    categoryColList = data.select_dtypes(item.type.__name__).columns.to_list()
                    enc = OrdinalEncoder()
                    data[categoryColList] = enc.fit_transform(data[categoryColList])
                    return data
            return None

    ret = dfEncodeDeal(data, calType=outlierFilterMethod)
    if (type(ret) != None):
        data = ret

    # Feature scaling (normalization)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(data)
    data.loc[:, :] = scaled_values

    # Outlier Removal
    #     clf = IsolationForest(contamination=.1)
    #     y_pred_train = clf.fit_predict(data.values)
    #     data = data[np.where(y_pred_train == 1, True, False)]
    #     targetColSeries = targetColSeries.filter(items=list(data.index.values),axis=0)

    return (data, targetColSeries)


ret = trainProcess(trainData, 'AMT_INCOME_TOTAL', 'ordinal')
trainData_regression = ret[0]
targetColSeries_regression = ret[1]

ret = trainProcess(trainData, 'TARGET', 'oneHot')
trainData_classification = ret[0]
targetColSeries_classification = ret[1]

''' 
    Feature selection 
'''
# Variance filtering in classification
vt = VarianceThreshold(threshold=0)  # np.percentile(trainData_classification.var().values, 80)
trainData_classification_varfilter = pd.DataFrame(vt.fit_transform(trainData_classification),
                                                  columns=trainData_classification.columns[vt.get_support()])

# Variance filtering in regression
vt = VarianceThreshold(threshold=0)  # np.percentile(trainData_regression.var().values, 80)
trainData_regression_varfilter = pd.DataFrame(vt.fit_transform(trainData_regression),
                                              columns=trainData_regression.columns[vt.get_support()])

# Correlation Filtering (MIC) in classification
# result = mutual_info_classif(trainData_classification_varfilter,targetColSeries_classification)
# col = trainData_classification_varfilter.columns.values.tolist()
# trainData_classification_correlationFilter = trainData_classification_varfilter[list(compress(col, list(result!=0)))]

# Correlation Filtering (MIC) in regression
# result = mutual_info_regression(trainData_regression_varfilter,targetColSeries_regression)
# col = trainData_regression_varfilter.columns.values.tolist()
# trainData_regression_correlationFilter = trainData_regression_varfilter[list(compress(col, list(result!=0)))]


''' 
    testData preprocessing
'''
# Separate continuous and discrete data
testData_continuous = testData.loc[:,
                      [dtype.name != 'object' for dtype in testData.dtypes.values.tolist()]]  # continuous data
testData_discrete = testData.loc[:,
                    [dtype.name == 'object' for dtype in testData.dtypes.values.tolist()]]  # discrete data
# For missing values of continuous variables, fill in with the mean
if (testData_continuous.shape[1] != 0):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    testData_fillNan_continuous = pd.DataFrame(imp.fit_transform(testData_continuous),
                                               columns=testData_continuous.columns)
else:
    testData_fillNan_continuous = testData_continuous
# For missing values of discrete features, treat the missing values as a separate category
if (testData_discrete.shape[1] != 0):
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='NA')
    testData_fillNan_discrete = pd.DataFrame(imp.fit_transform(testData_discrete), columns=testData_discrete.columns)
else:
    testData_fillNan_discrete = testData_discrete
# concat
testData_dealNan = pd.concat([testData_fillNan_continuous, testData_fillNan_discrete], axis=1)


def encodeDeal(targetCol, calType):
    targetColSeries = testData_dealNan[targetCol]
    data = testData_dealNan.drop([targetCol], axis=1)

    def dfEncodeDeal(data, calType='oneHot'):
        if (calType == 'oneHot'):
            for item in data.dtypes.unique():
                if (item.type.__name__ == 'object_'):
                    categoryColList = data.select_dtypes(item.type.__name__).columns.to_list()
                    #                     print('####################Print discrete value ranges of categorical features：####################')
                    #                     for col in categoryColList:
                    #                         print(data[col].unique())
                    df_onehot = pd.get_dummies(data[categoryColList])
                    data.drop(categoryColList, axis=1, inplace=True)
                    data = pd.concat([data, df_onehot], axis=1)
                    return data
            return None
        elif (calType == 'ordinal'):
            for item in data.dtypes.unique():
                if (item.type.__name__ == 'object_'):
                    categoryColList = data.select_dtypes(item.type.__name__).columns.to_list()
                    enc = OrdinalEncoder()
                    data[categoryColList] = enc.fit_transform(data[categoryColList])
                    return data
            return None

    ret = dfEncodeDeal(data, calType=calType)
    if (type(ret) != None):
        data = ret
    return (data, targetColSeries)


ret = encodeDeal('TARGET', 'oneHot')
testData_classification_data = ret[0]
testData_classification_target = ret[1]

ret = encodeDeal('AMT_INCOME_TOTAL', 'ordinal')
testData_regression_data = ret[0]
testData_regression_target = ret[1]


# Feature scaling (normalization)
def scale(data):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(data)
    data.loc[:, :] = scaled_values
    return data


testData_classification_data = scale(testData_classification_data)
testData_regression_data = scale(testData_regression_data)

'''
    Classification
'''
# Training set
X_train = trainData_classification_varfilter
y_train = targetColSeries_classification

# test set
X_test = testData_classification_data
y_test = testData_classification_target

# Ensure that the dimensions of the training set and the test set are the same
features = X_train.columns.values.tolist()
features = list(set(features).intersection(set(X_test.columns.values.tolist())))
X_train = X_train[features]
X_test = X_test[features]

model = LogisticRegression(penalty='l2')
model.fit(X_train, y_train)
prediction = model.predict(X_test)
report = classification_report(y_test, prediction, output_dict=True)
df_evaluate = pd.DataFrame(
    [["5330254", report["macro avg"]["precision"], report["macro avg"]["recall"], report["accuracy"]]],
    columns=["zid", "average_precision", "average_recall", "accuracy"])
df_evaluate.to_csv('z5330254.PART2.summary.csv', index=False)

df_predict = pd.concat([testData.SK_ID_CURR, pd.Series(prediction)], axis=1)
df_predict.columns = ['SK_ID_CURR', 'predicted_target']
df_predict.to_csv('z5330254.PART2.output.csv', index=False)

'''
    Regression
'''
# Training set
X_train = trainData_regression_varfilter
y_train = targetColSeries_regression
y_trainMean = y_train.mean()
y_trainStd = y_train.std()
y_train = y_train.apply(lambda x: (x - y_trainMean) / y_trainStd)

# test set
X_test = testData_regression_data
y_test = testData_regression_target
y_testMean = y_test.mean()
y_testStd = y_test.std()
y_test = y_test.apply(lambda x: (x - y_trainMean) / y_trainStd)

# Ensure that the dimensions of the training set and the test set are the same
features = X_train.columns.values.tolist()
features = list(set(features).intersection(set(X_test.columns.values.tolist())))
X_train = X_train[features]
X_test = X_test[features]

ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
ridgecv.fit(X_train, y_train)
model = Ridge(alpha=ridgecv.alpha_)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
df_evaluate = pd.DataFrame([["5330254", mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]]],
                           columns=["zid", "MSE", "correlation"])
df_evaluate.to_csv('z5330254.PART1.summary.csv', index=False)

df_predict = pd.concat(
    [testData.SK_ID_CURR, pd.Series(prediction.tolist()).apply(lambda x: x * y_testStd + y_testMean)], axis=1)
df_predict.columns = ['SK_ID_CURR', 'predicted_income']
df_predict.to_csv('z5330254.PART1.output.csv', index=False)

end = time.time()
print("Program execution time {0} second".format(end - start))
