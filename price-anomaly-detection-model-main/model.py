#%%
import clickhouse2pandas as ch2pd
import pandas as pd
import numpy as np
import os
import seaborn as sns
import urllib
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from sklearn.cluster import KMeans
import datetime
import pandahouse as ph
from sqlalchemy import create_engine
import pyodbc

verticals = {"HOME": "'HW', 'MA', 'TC', 'TS'",
             "ELECTRONICS": "'AC', 'AV', 'PA', 'IT','MO', 'HA'", 
             "FMCG": "'BC', 'DF', 'FF', 'HC', 'PC'", 
             "FASHION": "'AX', 'FW', 'RW', 'SW', 'KW'" }

anomalies = dict()

for vertical in verticals.keys():

    print('vertical : ' + vertical)
    print('exporting data from CLICKHOUSE')
    connection_url = 'enter your own connection'
    query = ''' select *
                from dk_warehouse.anomaly_detection
                where supplyCat in (''' + verticals[vertical] + ");"
    
    df = ch2pd.select(connection_url, query)

    df = df.astype({'selling_price':'float','rrp_price':'float', 'leafCatID':'str', 'brandID':'str'})
    df = df[df['rrp_price'] != 0]
    df['log_selling_price'] = np.log(df['rrp_price'])
    df['leafbrand'] = df['leafCatID'] + "|" + df['brandID']
    df['Last_cluster'] = False

    print('\tleafcategory clustering')
    
    def k_selection(category):
        global df
        X = np.array(df[df['leafCatID'] == category]['log_selling_price'])
        X = np.reshape(X, (len(X),1))
        if len(np.unique(X)) < 10:
            return 1
        else:
            min_value = np.inf
            best_k = 1
            for k in range(1,5):
                kmeans = KMeans(n_clusters = k)
                kmeans.fit(X)
                if k + np.log(-kmeans.score(X)) < min_value:
                    best_k = k
                    min_value = k + np.log(-kmeans.score(X))
            return best_k
    
    for category in df['leafCatID'].unique():
        k = k_selection(category)
        if k > 1:
            X = np.array(df[df['leafCatID'] == category]['log_selling_price'])
            X = np.reshape(X, (len(X),1))
            kmeans = KMeans(n_clusters = k, init='k-means++')
            kmeans.fit(X)
            labels = pd.DataFrame(kmeans.labels_)
            labels.rename(columns = {0:'cluster_number'}, inplace= True)
            centroids = kmeans.cluster_centers_
            imax1 = np.where(centroids == np.sort(centroids.reshape(k))[-1])[0][0]
            imax2 = np.where(centroids == np.sort(centroids.reshape(k))[-2])[0][0]
            labels.set_index(df[df['leafCatID'] == category]['log_selling_price'].index, inplace=True)
            if len(labels[labels['cluster_number'] == imax1]) < 10:
                Last_cluster = labels[(labels['cluster_number'] == imax1) | (labels['cluster_number'] == imax2)]
            Last_cluster = labels[(labels['cluster_number'] == imax1)]
            df.at[Last_cluster.index,'Last_cluster'] = True
        else :
            df.at[df[df['leafCatID'] ==  category].index.values,'Last_cluster'] = True
    
    print('\tcalculating statistical features')
    
    leafcat_cluster_whiskers = dict()
    print('\t\twhishi - leafCatID')
    for i in df['leafCatID'].unique():
        leafcat_cluster_whiskers[i] = boxplot_stats(df[(df['leafCatID'] == i)&(df['Last_cluster'] == True)]['log_selling_price']).pop(0)['whishi']
    leafbrand_whiskers = dict()
    
    print('\t\twhishi - leafbrand')
    for i in df['leafbrand'].unique():
        leafbrand_whiskers[i] = boxplot_stats(df[df['leafbrand'] == i]['log_selling_price']).pop(0)['whishi']
    m = 3
    leafcat_cluster_out_of_std = dict()
    
    print('\t\tnormal dist - leafCatID')
    for i in df['leafCatID'].unique():
        leafcat_cluster_out_of_std[i] = np.mean(df[(df['leafCatID'] == i)&(df['Last_cluster'] == True)]['log_selling_price']) 
        leafcat_cluster_out_of_std[i] += m * np.std(df[(df['leafCatID'] == i)&(df['Last_cluster'] == True)]['log_selling_price'])
    leafbrand_out_of_std = dict()
    
    print('\t\tnormal dist - leafbrand')
    for i in df['leafbrand'].unique():
        leafbrand_out_of_std[i] = np.mean(df[df['leafbrand'] == i]['log_selling_price']) + m * np.std(df[df['leafbrand'] == i]['log_selling_price'])
    
    df['leafcat_cluster_whiskers'] = df['leafCatID'].apply(lambda x: leafcat_cluster_whiskers[x])
    df['leafbrand_whiskers'] = df['leafbrand'].apply(lambda x: leafbrand_whiskers[x])
    df['leafcat_cluster_out_of_std'] = df['leafCatID'].apply(lambda x: leafcat_cluster_out_of_std[x])
    df['leafbrand_out_of_std'] = df['leafbrand'].apply(lambda x: leafbrand_out_of_std[x])

    def rutn_bool2number(var):
        if var:
            return 1
        return 0
    print('\tcalculating possibility features')
    df['possibility'] = (df['leafcat_cluster_whiskers'] < df['log_selling_price']).apply(lambda x:rutn_bool2number(x))
    df['possibility'] += (df['leafcat_cluster_out_of_std'] < df['log_selling_price']).apply(lambda x:rutn_bool2number(x))
    df['possibility'] += (df['leafbrand_out_of_std'] < df['log_selling_price']).apply(lambda x:rutn_bool2number(x))
    df['possibility'] += (df['leafbrand_whiskers'] < df['log_selling_price']).apply(lambda x:rutn_bool2number(x))
    df['possibility'] /= 4

    df['day_lag'] = (datetime.datetime.today()-df['created_at']).astype('timedelta64[D]')
    anomaly_df = df[(df['possibility']>0)&(df['is_live']==1)&(df['day_lag']<30)&(df['productPriceType'] != "printed")]
    anomaly_df.loc[:,'detected_at'] = datetime.date.today()
    anomaly_df.drop(columns=[
      'log_selling_price',
      'leafbrand',
      'Last_cluster',
      'leafcat_cluster_whiskers',
      'leafbrand_whiskers',
      'leafcat_cluster_out_of_std',
      'leafbrand_out_of_std',
      'day_lag'
    ], inplace=True)

    print('\tsaving ' + vertical + "'s data")
    anomalies[vertical] = anomaly_df

DF = pd.DataFrame()
for i in anomalies.keys():
    DF = pd.concat([DF,anomalies[i]])

data = DF.rename(columns = {
  'brandID':'brand_id',
  'brandNameFa':'brand_name_fa',
  'leafCatID':'leaf_cat_id',
  'leafCatName':'leaf_cat_name_fa',
  'supplyCat':'parent_supply_cat',
  'cat_lvl2_id':'supply_cat_id',
  'productPriceType':'product_price_type'
})

engine = create_engine('enter your own connection')
conn = engine.connect()

data.to_sql('detected_anomalies', conn, index = False, if_exists = 'append')
# %%
