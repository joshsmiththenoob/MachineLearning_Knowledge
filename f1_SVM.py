import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def scale_dataset(dataframe, oversample = False):
    X = dataframe[dataframe.columns[:-1]].values
    # in this time, Y will be 1-D array
    Y = dataframe[dataframe.columns[-1]].values
    
    sc = StandardScaler()
    # Scale the Known(Train) Data into range of 0 to 1
    X = sc.fit_transform(X)

    # Data Augumentation of less classes -> rebalance nums of dataset
    if oversample:
        ros = RandomOverSampler()
        X,Y = ros.fit_resample(X,Y)


    data = np.hstack((X,np.reshape(Y, (-1,1))))

    return data, X,Y

# make column name
cols = ['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist','class']

df = pd.read_csv(r'.\magic04.data', names = cols)

print(df.head())
# Y = df.iloc[:, -1]
# print(Y)

# # Label Encode : turn the text feature or text label into a digi-number feature
# le = LabelEncoder()
# Y = le.fit_transform(Y)
# df[-1] = Y
# print(Y,type(Y[8]))
# or Use this pandas preprocessing
df["class"] = (df["class"] == "g").astype(int) # True -> 1 , False -> 0 

# # Analysis the features in class : gamma, hadron
# for label in cols[:-1]:
#    # histogram of every feature of records about class == 1 (gamma)
#    print(df[df["class"]==1][label])
#    plt.hist(df[df["class"]==1][label], color='blue', label = 'gamma', alpha = 0.7, density= True)
#    plt.hist(df[df["class"]==0][label], color='red', label = 'hydron', alpha = 0.7, density= True)
#    plt.title(label)
#    plt.ylabel('Probability(Normalized Num)')
#    plt.legend()
#    plt.show()

# Create Train, Validation, Test datasets
# Shuffle data set(frac = 1) -> Split : Train : 0 -> 0.6 , Valid : 0.6 -> 0.8, Test : 0.8 -> 1
train, valid, test = np.split(df.sample(frac = 1),[int(0.6*len(df)),int(0.8*len(df))])
print(len(train))

# Scale all Features and Rebalance nums of data
train, X_train, Y_train = scale_dataset(train,oversample = True)

# the Unknown data (Don't Need to Rebalance nums of data)
valid, X_valid, Y_valid = scale_dataset(valid,oversample = False)
test, X_test, Y_test = scale_dataset(test,oversample = False)

# Using SVM Classifier Model : SVC
svc = SVC()
svc = svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)
print(classification_report(Y_test,Y_pred))