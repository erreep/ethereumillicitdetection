# -*- coding: utf-8 -*-
#import dependencies
import pandas as pd
from datetime import datetime
import requests
import pickle
import numpy as np
from shroomdk import ShroomDK

from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.model_selection import train_test_split
#from google.colab import drive

from google.colab import drive
drive.mount('/content/drive')

#google cloud credentials
credentials = service_account.Credentials.from_service_account_file(
'/content/drive/MyDrive/fill_in_credential_location.json')

project_id = 'fill_in_projectid'
client = bigquery.Client(credentials= credentials,project=project_id)

#flipsidecrypto credentials
API_KEY = "fill_in_flipside_api_key"

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
price=pd.read_csv('/content/drive/MyDrive/price.csv')
scam_transactions=vlookup(scam_transactions, price)
legal_transactions=vlookup(legal_transactions, price)
#feature engineering

scam_transactions = process_transactions(scam_transactions, price)
scam_transactions['scam'] = 1
legal_transactions = process_transactions(legal_transactions, price)

scam_transactions.to_csv('/content/drive/MyDrive/scam_transactions_me.csv')
legal_transactions.to_csv('/content/drive/MyDrive/legal_transactions_me.csv')
