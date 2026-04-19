import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv(r"C:\Users\sahar\OneDrive\Desktop\ds and ml files\sales.csv",index_col=0)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

le=LabelEncoder()
df['ShelveLoc']=le.fit_transform(df['ShelveLoc'])
df['Urban']=le.fit_transform(df['Urban'])
df['US']=le.fit_transform(df['US'])
df['high']=le.fit_transform(df['high'])

print(df.head())

X=df.drop('high',axis=1)
y=df['high']

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.6,
    max_features=0.6,
    random_state=42
)

model.fit(X_train,y_train)  

y_pred=model.predict(X_test)

print(model.score(X_test,y_test))


model_rf = RandomForestClassifier(
    n_estimators=80, 
    random_state=42,
    max_depth=10,
    max_samples=0.6,
    max_features=0.6
)

model_rf.fit(X_train, y_train)
print(model_rf.score(X_test, y_test))

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 80, 100],
    'max_depth': [None, 10, 20],
    'max_samples': [0.6, 0.8],
    'max_features': [0.6, 0.8]
}   

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
)

grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

from sklearn.metrics import confusion_matrix
y_pred_rf = model_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Best Random Forest')
plt.show()

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model_rf, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))

from sklearn.metrics import roc_curve, auc
y_pred_proba = model_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(
    fpr, tpr, 
    color='blue', lw=2, 
    label='ROC curve (area = %0.2f)' % roc_auc
)

plt.plot(
    [0, 1], [0, 1], 
     color='red', lw=2, 
    linestyle='--'
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()






