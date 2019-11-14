import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import metrics
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def get_remote_dataset(URL,header=None):
    try:
        path = get_file(URL.split('/')[-1], origin=URL)
    except:
        print('Error downloading remote dataset.')
        raise
    return pd.read_csv(path, header=header)

def get_local_dataset(PATH, header=None):
    try:
        df = pd.read_csv(PATH,header=header)
    except:
        print('Error loading local dataset.')
        raise
    df.dropna(inplace=True,axis=1) # For now, just drop NA's (rows with missing values)
    print('Read {} rows.'.format(len(df)))
    return df

def set_KDD_columns(kdd_df):
    kdd_df.columns = [
        'duration',
        'protocol_type',
        'service',
        'flag',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations',
        'num_shells',
        'num_access_files',
        'num_outbound_cmds',
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'outcome',
        'difficulty_rating'
    ]
    return kdd_df

def set_bin_class(df):
    for i, row in df.iterrows():
        if row['outcome'].split('.')[0] != 'normal':
            df.at[i, 'outcome'] = 'anomaly'

def set_multi_class(df):
    ### THIS WILL ONLY WORK WITH THE ENTIRE KDD DATASET. WILL NOT WORK WITH NSL-KDD SET!!! ###
    ### THIS IS YOUR ONLY WARNING!!!###
    DOS_TYPES = ('back','land','neptune','pod','smurf','teardrop')
    U2R_TYPES = ('buffer_overflow','loadmodule','perl','rootkit')
    R2L_TYPES = ('ftp_write','guess_passwd','imap','multihop','phf','spy','warezclient','warezmaster')
    PROBE_TYPES = ('ipsweep','nmap','portsweep','satan')
            
    for i, row in df.iterrows():
        val = 'normal'
        old_val = row['outcome'].split('.')[0]
        if old_val in DOS_TYPES:
            val = 'dos'
        elif old_val in U2R_TYPES:
            val = 'u2r'
        elif old_val in R2L_TYPES:
            val = 'r2l'
        elif old_val in PROBE_TYPES:
            val = 'probe'
        df.at[i,'outcome'] = val 

def encode_zscore(df, name, mean=None, std_dev=None):
    '''Encode numeric values as zscore'''
    if mean == None:
        mean = df[name].mean()
    if std_dev == None:
        std_dev = df[name].std()
    df[name] = (df[name] - mean) / std_dev

def encode_text(df, name):
    '''Encode text values to binary dummy values (i.e. red,blue is [0,1] or [1,0])'''
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        new_name = f"{name}-{x}"
        df[new_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def reencode_dataset(df):
    '''Takes a KDD pandas dataframe and transforms the data by 
    changing numeric columns to zscore, and text columns to 
    dummy values'''

    encode_zscore(df, 'duration')
    encode_text(df, 'protocol_type')
    encode_text(df, 'service')
    encode_text(df, 'flag')
    encode_zscore(df, 'src_bytes')
    encode_zscore(df, 'dst_bytes')
    encode_text(df, 'land')
    encode_zscore(df, 'wrong_fragment')
    encode_zscore(df, 'urgent')
    encode_zscore(df, 'hot')
    encode_zscore(df, 'num_failed_logins')
    encode_text(df, 'logged_in')
    encode_zscore(df, 'num_compromised')
    encode_zscore(df, 'root_shell')
    encode_zscore(df, 'su_attempted')
    encode_zscore(df, 'num_root')
    encode_zscore(df, 'num_file_creations')
    encode_zscore(df, 'num_shells')
    encode_zscore(df, 'num_access_files')
    encode_zscore(df, 'num_outbound_cmds')
    encode_text(df, 'is_host_login')
    encode_text(df, 'is_guest_login')
    encode_zscore(df, 'count')
    encode_zscore(df, 'srv_count')
    encode_zscore(df, 'serror_rate')
    encode_zscore(df, 'srv_serror_rate')
    encode_zscore(df, 'rerror_rate')
    encode_zscore(df, 'srv_rerror_rate')
    encode_zscore(df, 'same_srv_rate')
    encode_zscore(df, 'diff_srv_rate')
    encode_zscore(df, 'srv_diff_host_rate')
    encode_zscore(df, 'dst_host_count')
    encode_zscore(df, 'dst_host_srv_count')
    encode_zscore(df, 'dst_host_same_srv_rate')
    encode_zscore(df, 'dst_host_diff_srv_rate')
    encode_zscore(df, 'dst_host_same_src_port_rate')
    encode_zscore(df, 'dst_host_srv_diff_host_rate')
    encode_zscore(df, 'dst_host_serror_rate')
    encode_zscore(df, 'dst_host_srv_serror_rate')
    encode_zscore(df, 'dst_host_rerror_rate')
    encode_zscore(df, 'dst_host_srv_rerror_rate')

    return df

def generate_training_set(df, num_outcomes):
    '''This doesn't work well right now. FIX ME!!!'''
    while True:
        df_train = df.sample(frac=0.1, replace=False)
        dummies = pd.get_dummies(df_train['outcome'])
        if len(dummies.columns) != num_outcomes:
            continue
        x_columns = df_train.columns.drop(['outcome','difficulty_rating'])
        x = df_train[x_columns].values
        y = dummies.values
        break

    print(df_train.groupby('outcome')['outcome'].count())
    print(df_train.head())
    print(df_train.columns)

    return x, y

def build_classifier(x, y, hidden_layers=[8], activation='relu', epochs=5, batch_size=50, verbose=1):
    def baseline_model():
        model = Sequential()
        for layer in hidden_layers:
            model.add(Dense(layer, input_dim=x.shape[1], activation=activation))
        model.add(Dense(y.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    estimator = KerasClassifier(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return estimator

def run_tf():
    ###acquire and process dataset
    df = get_local_dataset('./nslkdd/KDDTest+.txt')
    #df_train = get_local_dataset('./nslkdd/KDDTrain+.txt')
    df = set_KDD_columns(df)
    df = reencode_dataset(df)
    set_multi_class(df)
    df.dropna(inplace=True, axis=1)
    x, y = generate_training_set(df,df['outcome'].nunique())

    ###build MLP model
    estimator = build_classifier(x, y, hidden_layers=[32,32])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    ###train model
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    estimator.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=1,epochs=100)

    ###evaluate model
    pred = estimator.predict(x_test)
    y_eval = np.argmax(y_test, axis=1)
    print(y_eval)
    score = metrics.accuracy_score(y_eval, pred)
    print("Validation score: {}".format(score))

def run_sklearn():
    df = get_local_dataset('./nslkdd/KDDTrain+.txt')
    df = set_KDD_columns(df)
    df = reencode_dataset(df)
    df.dropna(inplace=True, axis=1)
    x, y = generate_training_set(df,df['outcome'].nunique())

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(64,64,),max_iter=50,verbose=True)
    mlp.fit(x_train, y_train)

    predictions = mlp.predict(x_test)
    #print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test,predictions))

if __name__ == '__main__':
    #run_sklearn()
    run_tf()