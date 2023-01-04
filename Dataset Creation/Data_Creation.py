# -*- coding: utf-8 -*-

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
#from google.colab import drive

pip install shroomdk

from google.colab import drive
drive.mount('/content/drive')

#google cloud credentials
credentials = service_account.Credentials.from_service_account_file(
'/content/drive/MyDrive/google_cloud_credentials.json')

project_id = 'vproject-name-123456'
client = bigquery.Client(credentials= credentials,project=project_id)

#flipsidecrypto credentials
API_KEY = "API0-Key1-From2-Flipside3-Crypto4"

from shroomdk import ShroomDK
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

def get_scam_transactions(addresses):

    # Flatten the list of addresses
    addresses = [item[0] if isinstance(item, list) else item for item in addresses]

    # Execute the first query
    query_job = client.query("""
       SELECT t1.hash,t1.transaction_index, t1.from_address, t1.to_address, t1.value, t1.gas, t1.gas_price, t1.block_timestamp, t1.block_number 
       FROM bigquery-public-data.ethereum_blockchain.transactions t1
    WHERE receipt_status= 1  AND input='0x' AND to_address IN UNNEST(%s)
    """ %(addresses))
    scam_to = query_job.to_dataframe()

    # Execute the second query
    query_job = client.query("""
       SELECT t1.hash,t1.transaction_index, t1.from_address, t1.to_address, t1.value, t1.gas, t1.gas_price, t1.block_timestamp, t1.block_number 
       FROM bigquery-public-data.ethereum_blockchain.transactions t1
    WHERE receipt_status= 1  AND input='0x' AND from_address IN UNNEST(%s)
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
unique_scam_addresses = get_unique_addresses(A, B)
print('total scam adresses', len(unique_scam_addresses))
scam_transactions=get_scam_transactions(unique_scam_addresses)
scam_transactions=convert_timestamps(scam_transactions)
print(scam_transactions['block_number'].min())
print(scam_transactions['block_number'].max())
query_job = client.query("""
   SELECT t1.hash,t1.transaction_index, t1.from_address, t1.to_address, t1.value, t1.gas, t1.gas_price, t1.block_timestamp, t1.block_number FROM bigquery-public-data.ethereum_blockchain.transactions t1
WHERE receipt_status=1 and input='0x' and block_number BETWEEN 4370008 AND 9193145
ORDER BY RAND()
LIMIT 2000000""")
legal_transactions = query_job.to_dataframe()
#
legal_transactions = legal_transactions[~legal_transactions['hash'].isin(scam_transactions['hash'])]
legal_transactions=convert_timestamps(legal_transactions)
legal_transactions['scam']=0
scam_transactions['scam']=1
price=pd.read_csv('/content/drive/MyDrive/Thesis/2015-08-07_2022-12-23_ethereumprice_org.csv')
scam_transactions=vlookup(scam_transactions, price)
legal_transactions=vlookup(legal_transactions, price)
#feature engineering

scam_transactions = process_transactions(scam_transactions, price)
scam_transactions['scam'] = 1
legal_transactions = process_transactions(legal_transactions, price)

A = fetch_addresses_flipside(API_KEY)
B = fetch_addresses_cryptoscamdb()
unique_scam_addresses = get_unique_addresses(A, B)

scam_transactions.to_csv('/content/drive/MyDrive/scam_transactions_me.csv', index=False)
legal_transactions.to_csv('/content/drive/MyDrive/legal_transactions_me.csv', index=False))

scam_transactions=pd.read_csv('/content/drive/MyDrive/scam_transactions_me.csv')
legal_transactions=pd.read_csv('/content/drive/MyDrive/legal_transactions_me.csv')
price=pd.read_csv('/content/drive/MyDrive/2015-08-07_2022-12-23_ethereumprice_org.csv')

#outlier analysis
scam_transactions['from_address'].value_counts()

# Calculate the z-scores of each name in the 'names' column
name_counts_from = scam_transactions['from_address'].value_counts()
z_scores = stats.zscore(name_counts_from)

# Find the names with a z-score less than -3 or greater than 3
outlier_names_from = name_counts_from[(z_scores < -20) | (z_scores > 20)]

print(outlier_names_from)

scam_transactions = scam_transactions[~scam_transactions['from_address'].isin(outlier_names_from.index)]



import pandas as pd

ax=name_counts_from.plot.box()
ax.set_yticks(range(0, 110000, 10000))

# Calculate the z-scores of each name in the 'names' column
name_counts_to = scam_transactions['to_address'].value_counts()
z_scores = stats.zscore(name_counts_to)

# Find the names with a z-score less than -3 or greater than 3
outlier_names_to = name_counts_to[(z_scores < -20) | (z_scores > 20)]

print(outlier_names_to)

import pandas as pd

ax=name_counts_to.plot.box()


# Set the tick locations and labels for the y-axis
ax.set_yticks(range(0, 1100001, 100000))

scam_transactions = scam_transactions[~scam_transactions['from_address'].isin(outlier_names_from.index)]
scam_transactions = scam_transactions[~scam_transactions['to_address'].isin(outlier_names_to.index)]

scam_transactions['from_address'].value_counts()

import pandas as pd

# Calculate the lower and upper quartiles of the 'names' column
name_counts = scam_transactions['to_address'].value_counts()
lower_quartile = name_counts.quantile(0.25)
upper_quartile = name_counts.quantile(0.75)

# Find the names that fall below the lower quartile or above the upper quartile
outlier_names = name_counts[(name_counts < lower_quartile) | (name_counts > upper_quartile)]

print(outlier_names)

scam_transactions['to_address'].value_counts()

import random
random.seed(1)
random.shuffle(unique_scam_addresses)
num_scams = len(unique_scam_addresses)
split_1 = int(0.7 * num_scams)
split_2 = int(0.2 * num_scams)+split_1

group_train = unique_scam_addresses[:split_1]
group_val = unique_scam_addresses[split_1:split_2]
group_test = unique_scam_addresses[split_2:]

print('train', len(group_train)/len(unique_scam_addresses))
print('val', len(group_val)/len(unique_scam_addresses))
print('test', len(group_test)/len(unique_scam_addresses))

mask_train = scam_transactions['from_address'].isin(group_train) | scam_transactions['to_address'].isin(group_train)
mask_val = scam_transactions['from_address'].isin(group_val) | scam_transactions['to_address'].isin(group_val) 
mask_test = scam_transactions['from_address'].isin(group_test) | scam_transactions['to_address'].isin(group_test)

train_df=scam_transactions[mask_train]
val_df=scam_transactions[mask_val]
test_df=scam_transactions[mask_test]

print(len(scam_transactions))
train_proportion=len(train_df)/len(scam_transactions)
print(len(train_df), '', len(train_df)/len(scam_transactions))
val_proportion=len(val_df)/len(scam_transactions)
print(len(val_df), '', val_proportion)
test_proportion=len(test_df)/len(scam_transactions)
print(len(test_df), '', test_proportion)
print(len(val_df)+len(train_df)+len(test_df))

train_val_proportion=val_proportion+train_proportion
val_proportion_2=val_proportion/train_val_proportion
train_proportion_2=train_proportion/train_val_proportion

legal_train_num=len(legal_transactions)*train_proportion_2
val_proportion_2

legal_transactions_train = legal_transactions.sample(frac=train_proportion_2, random_state=42)

# Create df3 by removing the rows in df2 from df1
legal_transactions_val = legal_transactions.drop(legal_transactions_train.index)

# Concatenate the DataFrames

train_df = pd.concat([legal_transactions_train, train_df])
val_df = pd.concat([legal_transactions_val, val_df])
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
# Shuffle the indices
train_df = train_df.sample(frac=1).reset_index(drop=True)
val_df= val_df.sample(frac=1).reset_index(drop=True)

train_df.to_csv('/content/drive/MyDrive/train_df_final.csv', index=False)
val_df.to_csv('/content/drive/MyDrive/val_df_final.csv', index=False)

block_numbers=list(test_df['block_number'].unique())

query_job = client.query("""SELECT t1.hash,t1.transaction_index, t1.from_address, t1.to_address, t1.value, t1.gas, t1.gas_price, t1.block_timestamp, t1.block_number 
       FROM bigquery-public-data.ethereum_blockchain.transactions t1
    WHERE receipt_status= 1  AND input='0x' AND block_number IN UNNEST(%s)
    """ %(block_numbers))
block_transactions = query_job.to_dataframe()

indices_to_drop = block_transactions[block_transactions['hash'].isin(legal_transactions['hash'])].index
block_transactions.drop(index=indices_to_drop, inplace=True)
indices_to_drop=block_transactions[block_transactions['hash'].isin(train_df['hash'])].index
block_transactions.drop(index=indices_to_drop, inplace=True)
indices_to_drop=block_transactions[block_transactions['hash'].isin(val_df['hash'])].index
block_transactions.drop(index=indices_to_drop, inplace=True)

block_transactions=convert_timestamps(block_transactions)
block_transactions = block_transactions[~block_transactions['hash'].isin(scam_transactions['hash'])]
block_transactions['scam']=0
block_transactions=vlookup(block_transactions, price)
block_transactions = process_transactions(block_transactions, price)

test_df = pd.concat([block_transactions, test_df])
test_df = test_df.reset_index(drop=True)
# Shuffle the indices
test_df = test_df.sample(frac=1).reset_index(drop=True)

test_df.to_csv('/content/drive/MyDrive/test_df_final.csv', index=False)
