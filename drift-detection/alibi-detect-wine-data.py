# Package requirements
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

import alibi
from alibi_detect.cd import TabularDrift

# Split data
wine_data = load_wine()
feature_names = wine_data.feature_names
X, y = wine_data.data, wine_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50,
                                                    random_state=42)

# Initialize drift detector trigger when smaller
# difference occurs in data distribution with
# statisitcal significant test
cd = TabularDrift(x_ref=X_train, p_val=.05)

# Check drift for  statistical significant tests
preds = cd.predict(X_test)
labels = ['No', 'Yes']
print('Drift: {}'.format(labels[preds['data']['is_drift']]))

# Check drift at the level of features
fpreds = cd.predict(X_test + 4, drift_type='feature')
