

#import dependencies
import pandas as pd
from datetime import datetime
import requests
import pickle
import numpy as np
from shroomdk import ShroomDK
from scipy import stats
import random
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.model_selection import train_test_split
import requests
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib import pyplot
from xgboost import plot_importance
import xgboost as xgb
from catboost import CatBoostClassifier, Pool, cv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

pip install catboost

pip install catboost

from google.colab import drive
drive.mount('/content/drive')

#google cloud credentials
credentials = service_account.Credentials.from_service_account_file(
'/content/drive/MyDrive/Thesis/vivid-nomad-367714-60dd060ca378.json')

project_id = 'vivid-nomad-367714'
client = bigquery.Client(credentials= credentials,project=project_id)

#flipsidecrypto credentials
API_KEY = ""

def fetch_addresses_flipside(api_key):
    # Initialize `ShroomDK` with the provided API Key
    sdk = ShroomDK(api_key)

    # Parameters can be passed into SQL statements 
    # via native string interpolation
    sql = """
    SELECT
        *
    FROM ethereum.core.dim_labels
    WHERE label_subtype = 'toxic'
    """

    # Run the query against Flipside's query engine 
    # and await the results
    query_result_set = sdk.query(sql)

    # Create a Pandas dataframe from the query results
    df = pd.DataFrame(query_result_set.records)

    return df


def fetch_addresses_cryptoscamdb():
    response = requests.get("https://api.cryptoscamdb.org/v1/addresses")
    data = response.json()
    address_info_list = []
    for address in data['result']:
        address_info_list.append(data['result'][address])
    df = pd.DataFrame(address_info_list[0])
    for i in range(1, len(address_info_list)):
        df = pd.concat([df, pd.DataFrame(address_info_list[i])], axis=0)
    df = df[df.address.str.startswith("0x")]
    return df


def get_unique_addresses(A, B):
    A_df = A['address']
    B_df = B['address']
    A_addresses = A_df.tolist()
    B_addresses = B_df.tolist()
    addresses = A_addresses + B_addresses
    addresses = pd.DataFrame(addresses).drop_duplicates().values.tolist()
    addresses = [item[0] if isinstance(item, list) else item for item in addresses]
    return addresses

def get_transactions(addresses):

    # Flatten the list of addresses
    addresses = [item[0] if isinstance(item, list) else item for item in addresses]

    # Execute the first query
    query_job = client.query("""
       SELECT *
       FROM bigquery-public-data.ethereum_blockchain.transactions t1
    WHERE receipt_status= 1  AND input='0x' AND from_address is not null AND to_address is not null AND to_address IN UNNEST(%s)
    """ %(addresses))
    scam_to = query_job.to_dataframe()

    # Execute the second query
    query_job = client.query("""
       SELECT *
       FROM bigquery-public-data.ethereum_blockchain.transactions t1
    WHERE receipt_status= 1  AND input='0x' AND from_address is not null AND to_address is not null AND from_address IN UNNEST(%s)
    """ %(addresses))
    scam_from = query_job.to_dataframe()
    joined_df = pd.concat([scam_to, scam_from], axis=0, ignore_index=True)
    return joined_df

def get_min_max(df, column):
    # Find the minimum value of the column
    min_value = df[column].min()

    # Find the maximum value of the column
    max_value = df[column].max()

    return min_value, max_value

from datetime import datetime

def convert_timestamps(df):
  # Create a new column for the Unix timestamps
  df["timestamp"] = None

  # Iterate over the rows of the "block_timestamp" column
  for i, dt in df["block_timestamp"].iteritems():
    if isinstance(dt, datetime):
      # Convert the Timestamp object to a Unix timestamp and store it in the new column
      df.at[i, "timestamp"] = dt.timestamp()
    elif isinstance(dt, str):
      # Convert the datetime string to a datetime object
      dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S%z")

      # Convert the datetime object to a Unix timestamp and store it in the new column
      df.at[i, "timestamp"] = dt.timestamp()

  # Convert the timestamp column to integers
  df["timestamp"] = df["timestamp"].astype(int)

  # Create a new column for the daily Unix timestamps
  df["daily_unix_timestamp"] = None

  # Iterate over the rows of the "timestamp" column
  for i, ts in df["timestamp"].iteritems():
    # Convert the Unix timestamp to a datetime object
    dt = datetime.fromtimestamp(ts)

    # Get the date part of the datetime object
    date = dt.date()

    # Convert the date part back to a datetime object
    dt = datetime.combine(date, datetime.min.time())

    # Convert the datetime object to a Unix timestamp and store it in the new column
    df.at[i, "daily_unix_timestamp"] = dt.timestamp()

  # Convert the timestamp column to integers
  df["daily_unix_timestamp"] = df["daily_unix_timestamp"].astype(int)
  return df

def vlookup(df_new, df_price):
  # Merge the two DataFrames on the left_col and right_col columns
  df_merged = pd.merge(df_new, df_price, left_on='daily_unix_timestamp', right_on='timestamp')

  # Create a new column in df_new with the values from the "open" column in df_price
  values=list(df_merged["open"])
  df_new['price_eth'] = values
  return df_new

def gwei_to_eth(x):

  return x * 0.000000000000000001

def process_transactions(transactions, price):
  transactions = convert_timestamps(transactions)
  transactions['scam'] = 0
  transactions = vlookup(transactions, price)

  transactions["value"] = transactions["value"].astype(float)
  transactions["value_eth"] = transactions["value"].apply(gwei_to_eth)
  transactions["gas_price"] = transactions["gas_price"].astype(float)
  transactions["gas_price_eth"] = transactions["gas_price"].apply(gwei_to_eth)
  transactions['max_gas_cost_eth'] = transactions['gas'] * transactions['gas_price_eth']
  transactions["value_usd"] = transactions['value_eth'] * transactions['price_eth']
  transactions['max_gas_cost_eth'] = transactions['max_gas_cost_eth'] * transactions['price_eth']

  return transactions

A = fetch_addresses_flipside(API_KEY)
B = fetch_addresses_cryptoscamdb()
unique_illicit_addresses = get_unique_addresses(A, B)
print('total illicit adresses', len(unique_illicit_addresses))

illicit_transcations=pd.read_csv('/content/drive/MyDrive/Thesis/scam_transactions_me.csv')

A = fetch_addresses_flipside(API_KEY)
B = fetch_addresses_cryptoscamdb()
unique_illicit_addresses = get_unique_addresses(A, B)
print('total illicit adresses', len(unique_illicit_addresses))
illicit_transactions=get_transactions(unique_illicit_addresses)
illicit_transactions=convert_timestamps(illicit_transactions)
illicit_transactions['illicit']=1

print(illicit_transactions['block_number'].min())
print(illicit_transactions['block_number'].max())
query_job = client.query("""
   SELECT distinct(from_address) as address FROM bigquery-public-data.ethereum_blockchain.transactions t1
WHERE receipt_status=1 and input='0x' and block_number BETWEEN 4370008 AND 9193145
ORDER BY RAND()
LIMIT 1000000""")
nonillicit_accounts = query_job.to_dataframe()

#nonillicit_accounts.to_csv('/content/drive/MyDrive/Thesis/non_illicit_accounts.csv', index=False)

legal_list_final=list(nonillicit_accounts['address'])
chunk_size = 20000
chunked_list = [legal_list_final[i:i + chunk_size] for i in range(0, len(legal_list_final), chunk_size)]
def first_chunk():
  for chunk in chunked_list[:1]:
    df=get_transactions(chunk)
  return df
non_illicit=first_chunk()
def other_chunks(df):
  for chunk in chunked_list:
    
    df=df.append(get_transactions(chunk))
    print('currently', len(df), 'transactions in dataset')
  return df
non_illicit_transactions=other_chunks(non_illicit)

#non_illicit_transactions.to_csv('/content/drive/MyDrive/Thesis/non_illicit_raw.csv', index=False)

#len(ewaja)

#non_illicit_transactions=pd.read_csv('/content/drive/MyDrive/Thesis/non_illicit_df_semi_semifinal.csv')

blocks_list = list(range(4370008, 9193145))
indices_to_drop = non_illicit_transactions[~non_illicit_transactions['block_number'].isin(blocks_list)].index
non_illicit_transactions.drop(index=indices_to_drop, inplace=True)
indices_to_drop=non_illicit_transactions[non_illicit_transactions['hash'].isin(list(illicit_transactions['hash']))].index
non_illicit_transactions.drop(index=indices_to_drop, inplace=True)
non_illicit_transactions=convert_timestamps(non_illicit_transactions)
non_illicit_transactions['illicit']=0

blocks_list = list(range(4370008, 9193145))
indices_to_drop = non_illicit_transactions[~non_illicit_transactions['block_number'].isin(blocks_list)].index
non_illicit_transactions.drop(index=indices_to_drop, inplace=True)
indices_to_drop=non_illicit_transactions[non_illicit_transactions['hash'].isin(list(illicit_transactions['hash']))].index
non_illicit_transactions.drop(index=indices_to_drop, inplace=True)

non_illicit_transactions=pd.read_csv('/content/drive/MyDrive/Thesis/non_illicit_raw2.csv')

nonillicit_accounts=pd.read_csv('/content/drive/MyDrive/Thesis/non_illicit_accounts.csv')

illicit_transactions=pd.read_csv('/content/drive/MyDrive/Thesis/illicit_df_first.csv')

legal_list_final=list(nonillicit_accounts['address'])

non_illicit_transactions

non_illicit_transactions.to_csv('/content/drive/MyDrive/Thesis/non_illicit_df_real.csv', index=False)

import matplotlib.pyplot as plt
import seaborn as sns
# Calculate the z-scores of each name in the 'names' column
name_counts_from = illicit_transactions['from_address'].value_counts()
z_scores = stats.zscore(name_counts_from)

# Find the names with a z-score less than -3 or greater than 3
outlier_names_from = name_counts_from[(z_scores < -20) | (z_scores > 20)]
print(outlier_names_from)
ax=sns.boxplot(name_counts_from, showfliers=True)
ax.set(xlabel='Transactions per address')
plt.show()

# Calculate the z-scores of each name in the 'names' column
name_counts_to = illicit_transactions['to_address'].value_counts()
z_scores = stats.zscore(name_counts_to)
outlier_names_to = name_counts_to[(z_scores < -10) | (z_scores > 10)]

print(outlier_names_to)

ax=sns.boxplot(name_counts_to, showfliers=True)
ax.set(xlabel='Transactions per address')
plt.show()

# Calculate the z-scores of each name in the 'names' column
name_counts_from_NI = non_illicit_transactions['from_address'].value_counts()
z_scores = stats.zscore(name_counts_from_NI)
outlier_names_from_NI = name_counts_from_NI[(z_scores < -20) | (z_scores > 20)]

print(outlier_names_from_NI)

ax=sns.boxplot(name_counts_from_NI, showfliers=True)
ax.set(xlabel='Transactions per address')
plt.show()

#
# Calculate the z-scores of each name in the 'names' column
name_counts_to_NI = non_illicit_transactions['to_address'].value_counts()
z_scores = stats.zscore(name_counts_to_NI)
outlier_names_to_NI = name_counts_to_NI[(z_scores < -20) | (z_scores > 20)]

print(outlier_names_to_NI)

ax=sns.boxplot(name_counts_to_NI, showfliers=True)
ax.set(xlabel='Transactions per address')
plt.show()

illicit_transactions = illicit_transactions[~illicit_transactions['from_address'].isin(outlier_names_from.index)]
illicit_transactions = illicit_transactions[~illicit_transactions['to_address'].isin(outlier_names_to.index)]
#non_illicit_transactions = non_illicit_transactions[~non_illicit_transactions['from_address'].isin(outlier_names_from_NI.index)]
#non_illicit_transactions = non_illicit_transactions[~non_illicit_transactions['to_address'].isin(outlier_names_to_NI.index)]

non_illicit_transactions=pd.read_csv('')

unique_illicit_addresses

import random
random.seed(11)
random.shuffle(unique_illicit_addresses)
num_scams = len(unique_illicit_addresses)
split = int(0.8 * num_scams)



split

group_train = unique_illicit_addresses[:split]
group_test = unique_illicit_addresses[split:]

print('train', len(group_train)/len(unique_illicit_addresses))
print('test', len(group_test)/len(unique_illicit_addresses))

mask_train = illicit_transactions['from_address'].isin(group_train) | illicit_transactions['to_address'].isin(group_train)
mask_test = illicit_transactions['from_address'].isin(group_test) | illicit_transactions['to_address'].isin(group_test)

train_df=illicit_transactions[mask_train]
test_df=illicit_transactions[mask_test]

print(len(illicit_transactions))
train_proportion=len(train_df)/len(illicit_transactions)
print(len(train_df), '', len(train_df)/len(illicit_transactions))
test_proportion=len(test_df)/len(illicit_transactions)
print(len(test_df), '', test_proportion)

random.seed(123)
random.shuffle(legal_list_final)
num_scams_NI = len(legal_list_final)
split = int(0.80 * num_scams_NI)
group_train_NI = legal_list_final[:split]
group_test_NI = legal_list_final[split:]

mask_train_NI = non_illicit_transactions['from_address'].isin(group_train_NI) | non_illicit_transactions['to_address'].isin(group_train_NI)
mask_testNI = non_illicit_transactions['from_address'].isin(group_test_NI) | non_illicit_transactions['to_address'].isin(group_test_NI)

L_train_df=non_illicit_transactions[mask_train_NI]
L_test_df=non_illicit_transactions[mask_testNI]

print(len(non_illicit_transactions))
train_proportion_L=len(L_train_df)/len(non_illicit_transactions)
print(len(L_train_df), '', len(L_train_df)/len(non_illicit_transactions))
test_proportion_L=len(L_test_df)/len(non_illicit_transactions)
print(len(L_test_df), '', test_proportion_L)

train_df3 = pd.concat([L_train_df, train_df])
train_df3 = train_df3.reset_index(drop=True)
# Shuffle the indices
train_df3 = train_df3.sample(frac=1).reset_index(drop=True)

test_df3 = pd.concat([L_test_df, test_df])
test_df3 = test_df3.reset_index(drop=True)
# Shuffle the indices
test_df3 = test_df3.sample(frac=1).reset_index(drop=True)

train_df3 = train_df3[~train_df3['hash'].isin(test_df3['hash'])]

train_df3['illicit'].value_counts()

test_df3['illicit'].value_counts()

test_df3.to_csv('/content/drive/MyDrive/Thesis/test_account_based_3.csv', index=False)
train_df3.to_csv('/content/drive/MyDrive/Thesis/train_account_based_3.csv', index=False)

test_df3.isnull().sum()

train_df3.isnull().sum()

# Impute a 0 for the missing values in the 'transaction_type' column
train_df3['transaction_type'].fillna(value=0, inplace=True)
test_df3['transaction_type'].fillna(value=0, inplace=True)
# Impute a 0 for the missing values in the 'max_Fee_Per_Gas' column
train_df3['max_Fee_Per_Gas'].fillna(value=train_df3['transaction_type'], inplace=True)
test_df3['max_Fee_Per_Gas'].fillna(value=test_df3['transaction_type'], inplace=True)
# Impute a 0 for the missing values in the 'max_Priority_Fee_Per_Gas' column
train_df3['max_Priority_Fee_Per_Gas'].fillna(value=train_df3['transaction_type'], inplace=True)
test_df3['max_Priority_Fee_Per_Gas'].fillna(value=test_df3['transaction_type'], inplace=True)

###vanaf hier
train=pd.read_csv('/content/drive/MyDrive/Thesis/train_account_based_3.csv')
test=pd.read_csv('/content/drive/MyDrive/Thesis/test_account_based_3.csv')
train=train.drop(['hash','from_address','to_address','receipt_contract_address','receipt_root', 'block_timestamp', 'block_hash', 'input','nonce'],axis=1)
test=test.drop(['hash','from_address','to_address','receipt_contract_address','receipt_root', 'block_timestamp', 'block_hash','input','nonce'],axis=1)
X_train=train.drop(['illicit'],axis=1)
y_train=train['illicit']
X_test=test.drop(['illicit'],axis=1)
y_test=test['illicit']
#column value as float
X_train=X_train.astype(float)
X_test=X_test.astype(float)
X_train['value'] = np.log(X_train['value'].astype(float) + 0.0001)
X_test['value'] = np.log(X_test['value'].astype(float) + 0.0001)

df['value'] = np.log(df['value'])

df['valuelog'] = np.log(df['value'].astype(float) + 0.0001)

min(df['valuelog'])

df['value'] = df['value'].apply(lambda x: '{:.8f}'.format(x))

# Take the logarithm of the 'value' column
#df['value'] = np.log(df['value'].astype(float))

import seaborn as sns

# Create a boxplot of the 'value' column
sns.boxplot(x=df['gas_price'])
plt.xlabel('Value')
plt.title('Boxplot of Value')
plt.show()

import seaborn as sns

# Create a box plot of the 'value' column, grouped by the 'illicit' column
sns.boxplot(x='illicit', y='value', data=df)
plt.xlabel('Illicit')
plt.ylabel('Value')
plt.title('Box plot of Value grouped by Illicit')
plt.show()

#Model training

# Create a logistic regression model
log_reg = LogisticRegression()

# Fit the model to the training data
log_reg.fit(X_train, y_train)

# Predict on the test data
y_pred = log_reg.predict(X_test)
y_pred_train = log_reg.predict(X_train)
# Calculate the F1 score and AUC
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print("F1 score: ", f1)
print("AUC: ", auc)

# Create a confusion matrix
confusion = confusion_matrix(y_train, y_pred_train)

# Create a heatmap of the confusion matrix
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=["non-illicit", "illicit"], yticklabels=["non-illicit", "illicit"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#random forest

# Define the parameter grid for the Random Search
param_grid = {'n_estimators': randint(10, 100),
              'max_depth': randint(1, 10),
              'min_samples_split': randint(2, 10),
              'min_samples_leaf': randint(1, 10),}

# Initialize the Random Forest classifier
rf = RandomForestClassifier()

# Initialize the Random Search object
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, scoring=['f1', 'roc_auc'], refit='f1', cv=cross_val, random_state=42, verbose=3)

# Fit the Random Search object to the data
random_search.fit(X_train, y_train)

# Get the best estimator from the Random Search
best_estimator = random_search.best_estimator_

# Use the best estimator to make predictions on the test set
y_pred = best_estimator.predict(X_test)

#Catboost
cross_val = StratifiedKFold(n_splits=5)
# Define the parameter grid
param_grid = {
    'max_depth': [ 5, 8,10],
    'learning_rate': [ 0.1, 0.4, 0.5],
    'n_estimators': [100, 500,1000,3000],
    'scale_pos_weight':[0.1,1,10]
}

# Create the CatBoost model
model = cb.CatBoostClassifier()

# Use random search to tune the hyperparameters
random_search_Catboost = RandomizedSearchCV(model, param_grid, cv=5, n_iter=20, scoring=['f1', 'roc_auc'], refit='f1', verbose=3)

#XGBoosting
# Define the parameter grid
param_grid = {
    'max_depth': [ 5, 8,10],
    'learning_rate': [ 0.1, 0.4, 0.5],
    'n_estimators': [100, 500,1000],
    'scale_pos_weight':[0.1,1,10],
}

# Convert the data into an XGBoost-compatible format
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

# Create the XGBoost model
bst = xgb.XGBClassifier()

# Use random search to tune the hyperparameters
random_search = RandomizedSearchCV(bst, param_grid, cv=cross_val, n_iter=10, scoring=['f1', 'roc_auc'], refit='f1', random_state=42, verbose=3)
all_results_XGB = pd.DataFrame(random_search.cv_results_)
#all_results5.to_csv('all_results_xgboostthesisdone.csv')
bestparamsXGB=all_results_XGB[all_results_XGB['rank_test_f1']==1]['params']
BestresultsXGB=all_results_XGB[all_results_XGB['rank_test_f1']==1]
for i in bestparamsXGB:
    print(i)

#training with best and then predicting on test

#Logistic regression

#Random forest
# Create the model object
modelrf = RandomForestClassifier(max_depth=9, min_samples_leaf=3, min_samples_split=6, n_estimators=60)

# Fit the model to your training data
modelrf.fit(X_train, y_train)
yhat_rf=modelrf.predrict(y_test)
#f1 score
print('f1 score is:', f1_score(y_test, yhat_rf, average='macro'))
#AUC 
print('AUC is:', roc_auc_score(y_test, yhat_rf))

plot_importance(modelrf)
plt.show()

#Catboost
modelCboost = CatBoostClassifier(learning_rate=0.5, max_depth=10, n_estimators=500, custom_loss=['AUC'])

# Fit the model on the training data
modelCboost.fit(X_train, y_train,  verbose=3)

y_hat=modelCboost.predict(X_test)
confusion = confusion_matrix(y_test, y_hat)
# Create a heatmap of the confusion matrix
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=["non-illicit", "illicit"], yticklabels=["non-illicit", "illicit"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#f1 score
print('f1 score is:', f1_score(y_test, y_hat, average='macro'))
#AUC 
print('AUC is:', roc_auc_score(y_test, y_hat))

plot_importance(modelCboost)
plt.show()

#XGboost
# Create the XGBoost model

XGBmodel = xgb.XGBClassifier(learning_rate=0.4, max_depth=10, n_estimators=1000, scale_pos_weight=1, eval_metric = 'auc')
XGBmodel.fit(X_train, y_train, verbose=3)
y_hat=XGBmodel.predict(X_test)
confusion = confusion_matrix(y_test, y_hat)

# Create a heatmap of the confusion matrix
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=["non-illicit", "illicit"], yticklabels=["non-illicit", "illicit"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#f1 score
print('f1 score is:', f1_score(y_test, y_hat, average='macro'))
#AUC 
print('AUC is:', roc_auc_score(y_test, y_hat))

plot_importance(XGBmodel)
plt.show()

df=pd.concat([train,test])

from scipy import stats
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assume your data is stored in a pandas DataFrame called 'df'

# Split the data into two groups: illicit and non-illicit transactions
illicit_transactions = df[df['illicit'] == 1]
non_illicit_transactions = df[df['illicit'] == 0]

# Extract the 'value' column for each group
illicit_values = illicit_transactions['value']
non_illicit_values = non_illicit_transactions['value']

# Conduct a t-test
t, p = stats.ttest_ind(illicit_values, non_illicit_values)
print(f'T-value: {t}, P-value: {p}')

import matplotlib.pyplot as plt
import seaborn as sns

# Create a new dataframe with the mean value for each group
mean_df = df.groupby('illicit')['gas_price'].mean().reset_index()

# Create a bar plot
sns.barplot(x='illicit', y='gas_price', data=mean_df)
plt.xlabel('Illicit')
plt.ylabel('Mean Value')
plt.title(' plot of Mean Value for Illicit and Non-Illicit Transactions')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Create a new dataframe with the mean value for each group
mean_df = df.groupby('illicit')['gas_price'].mean().reset_index()

# Create a bar plot
sns.barplot(x='illicit', y='gas_price', data=mean_df)
plt.xlabel('Illicit')
plt.ylabel('Mean gas')
plt.title('Bar plot of Mean Gas price  for Illicit and Non-Illicit Transactions')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Create a new dataframe with the mean value for each group
mean_df = df.groupby('illicit')['gas'].mean().reset_index()

# Create a bar plot
sns.barplot(x='illicit', y='gas', data=mean_df)
plt.xlabel('Illicit')
plt.ylabel('Mean gas')
plt.title('Bar plot of Mean Gas for Illicit and Non-Illicit Transactions')
plt.show()

# Create a box plot of the 'value' column, grouped by the 'illicit' column
sns.boxplot(x='illicit', y='value', data=df)

# Create a dictionary to map the values of the x-axis
map_dict = {0: 'Non-Illicit', 1: 'Illicit'}

# Use the 'set_xticklabels' method to change the labels of the x-axis
plt.gca().set_xticklabels([map_dict[i] for i in plt.gca().get_xticks()])

plt.xlabel('Illicit')
plt.ylabel('Value')
plt.title('Box plot of Value grouped by Illicit')
plt.show()

# Assume your data is stored in a pandas DataFrame called 'df'

# Convert the timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Extract the year from the timestamp column
df['year'] = df['timestamp'].dt.year

# Group the data by the 'year' column
grouped = df.groupby('year')

# Calculate the number of illicit transactions per year
illicit_per_year = grouped['illicit'].sum()


# Print the number of illicit transactions per year
print(illicit_per_year)

# Plot the number of illicit transactions per year
illicit_per_year.plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Number of Illicit Transactions')
plt.title('Number of Illicit Transactions per Year')
plt.show()

df['illicit'].value_counts()
