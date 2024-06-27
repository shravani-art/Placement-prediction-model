import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

placement_df=pd.read_csv("Campus_Selection.csv")
placement_df_required=placement_df.drop("sl_no",axis=1)
# placement_df_required.head()

status_placed=placement_df_required[placement_df_required['status']=='Placed']
status_not_placed=placement_df_required[placement_df_required['status']=='Not Placed']
# status_not_placed.shape

predictor_df=placement_df_required.drop('status',axis=1)
target_df=placement_df_required[['status']]

ros = RandomOverSampler(random_state=23)
x_ros, y_ros = ros.fit_resample(predictor_df, target_df)
# y_ros.value_counts()

enc=LabelEncoder()
y_ros['status_binary']=enc.fit_transform(y_ros['status'])

enc1=LabelEncoder()
x_ros['workex_binary']=enc1.fit_transform(x_ros['workex'])
x_ros.drop('workex',axis=1,inplace=True)

ordinal_list=['Central','Others']
ct=ColumnTransformer([('ohe',OneHotEncoder(drop='first'),['gender', 'hsc_s', 'degree_t', 'specialisation']),
                     ('oe',OrdinalEncoder(categories=[ordinal_list,ordinal_list]),['ssc_b','hsc_b']),
                     ],remainder='passthrough')

x_encoded=ct.fit_transform(x_ros)

X_train,X_test,y_train,y_test=train_test_split(x_encoded,y_ros[['status_binary']],test_size=0.30,random_state=15)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)

def pre_process_data(predictor_data):
    predictor_data['workex_binary']=enc1.transform(predictor_data['workex'])
    predictor_data_new=predictor_data.drop('workex',axis=1)
    predictor_data_encoded=ct.transform(predictor_data_new)
    X_data_final = sc.transform(predictor_data_encoded) 
    return X_data_final
    