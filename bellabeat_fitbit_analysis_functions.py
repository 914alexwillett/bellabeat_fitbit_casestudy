import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
import secrets
import sqlite3
from collections import Counter
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
sns.set_style('whitegrid')

# SQL functions #
def sql_query(query_string):
    """Takes a SQL query is a string input and then returns the result"""
    # connect to the Database
    connection = sqlite3.connect('fitbit_database.db')
    cursor = connection.cursor()
    # execute the query
    cursor.execute(query_string)
    # return result
    query_result = cursor.fetchall()
    # close the connect to the Database
    cursor.close()
    return query_result

def query_to_df(query_string, col_names, Id_incl=False):
    query_result = sql_query(query_string)
    df = pd.DataFrame(query_result)
    df.columns = col_names
    if Id_incl:
        # turns Id column into a string turning the column into a label
        df['Id'] = df['Id'].astype('string')
    return df

def sql_insert(query_string):
    # connect to the Database
    connection = sqlite3.connect('fitbit_database.db')
    cursor = connection.cursor()
    # execute the query
    cursor.execute(query_string)
    # commit to
    # close the connect to the Database
    cursor.close()  
    
def sql_execute(query_string):
    # connect to the Database
    connection = sqlite3.connect('fitbit_database.db')
    cursor = connection.cursor()
    # execute the query
    cursor.execute(query_string)
    # close the connect to the Database
    cursor.close()  
    
def simple_sql_query(query_string):
    """Take a SQL query as a string input and then returns the single result"""
    # connect to the Database
    connection = sqlite3.connect('fitbit_database.db')
    cursor = connection.cursor()
    # execute the query
    cursor.execute(query_string)
    # return result
    query_result = cursor.fetchall()
    # close the connect to the Database
    cursor.close()
    return query_result[0][0]


# Analysis functions #
def calc_linear_regression(_x, _y):
    x, y = _x, _y
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    y_predict = []
    for item in x:
        y_predict.append((slope * item) + intercept)
    r2 = r2_score(y, y_predict)
    print('Slope:', slope)
    print('Intercept:', intercept)
    print('Pearson R-score:', r)
    print('P-value:', p) 
    print('Standard Error:', std_err)
    print('R-squared:', r2)
    plt.rcParams['figure.figsize'] = 9, 7 
    sns.regplot(x=x, y=y)
    return [slope, intercept, r, p, std_err]

def plot_corr_matrix(df):
    df_corr = df.corr()
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    f, ax = plt.subplots(figsize=(16, 16))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(df_corr, mask=mask, vmin=-1, vmax=1, cmap=cmap,
                square=True, annot=True)
    
def plot_daily_count(df, cols):
    cols_len = len(cols)
    # if even number of histogram pairs
    if cols_len % 2 == 0:
        for i in range(0, cols_len, 2):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            sns.histplot(data = df, x = df[cols[i]], ax=axes[0])
            axes[0].axvline(x=round(df[cols[i]].mean()), ls = '--', color='black')
            sns.histplot(data = df, x = df[cols[i+1]], ax=axes[1])
            axes[1].axvline(x=round(df[cols[i+1]].mean()), ls = '--', color='black')
        plt.show()
    else:
        for i in range(0, cols_len-1, 2):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            sns.histplot(data = df, x = df[cols[i]], ax=axes[0])
            axes[0].axvline(x=round(df[cols[i]].mean()), ls = '--', color='black')
            sns.histplot(data = df, x = df[cols[i+1]], ax=axes[1])
            axes[1].axvline(x=round(df[cols[i+1]].mean()), ls = '--', color='black')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.histplot(data = df, x = df[cols[cols_len-1]], ax=axes[0])
        axes[0].axvline(x=round(df[cols[cols_len-1]].mean(), 1), ls = '--', color='black')
        plt.show()
        
def plot_daily_hist(df, column):
    plt.rcParams['figure.figsize'] = 7, 7
    plt.rcParams.update({'font.size': 11})
    mean = round(df.aggregate({column: 'mean'})[0], 2)
    sns.histplot(data = df, x = column)
    title = f'Daily {column.capitalize()} Histogram'
    plt.axvline(x=mean, ls = '--', color='black')
    plt.figtext(0.8, 0.06, f'Mean: {mean}', ha="center", fontsize=11)
    plt.title(title)
    plt.show()
        
### Different implimentations
def plot_wkDay_stats(df, column, aggregate_type):
    df_wkDay = df.groupby('wkDay').aggregate({column:[aggregate_type]}).reset_index()
    col_name = f'{aggregate_type.capitalize()} {column}'
    df_wkDay.columns = ['wkDay', col_name]
    fig, ax = plt.subplots(figsize=(8, 8))
    xs = ['Mon', 'Tue', 'Wed', 'Th', 'Fri', 'Sat', 'Sun']
    ys = df_wkDay[col_name]
    fig = sns.barplot(x=xs, y=ys)
    title_str = f'{col_name} by Day of the Week'
    fig.set(xlabel='Day of the Week', ylabel=col_name, title=f'{col_name} by Day of the Week')
    plt.show()

def plot_query_barplot(query_string, col_names, Id_incl=False, mean_line=None, title=None):
    # can only take 2 variables for col_names
    df = query_to_df(query_string, col_names, Id_incl)
    plt.figure(figsize=(8, 6))
    sns.barplot(y=col_names[0], x=col_names[1], data = df)
    if mean_line != None:
        plt.axvline(mean_line, ls='--', color='black')
        plt.figtext(0.8, 0.06, f'Overall Mean: {mean_line}', ha="center", fontsize=11)
    if title:
        plt.title(title)
    plt.show()
    
def plot_query_barplot2(query_string, col_names, Id_incl=False, mean_line=None, title=None):
    # can only take 2 variables for col_names
    result = sql_query(query_string)
    df = pd.DataFrame(result)
    df.columns = col_names
    if Id_incl == True:
        df['Id'] = df['Id'].astype('string')
    plt.figure(figsize=(8, 6))
    sns.barplot(y=col_names[0], x=col_names[1], data = df)
    if mean_line != None:
        plt.axvline(mean_line, ls='--', color='black')
        plt.figtext(0.8, 0.06, f'Overall Mean: {mean_line}', ha="center", fontsize=11)
    if title:
        plt.title(title)
    plt.show()
    
def plot_hourly_heartrateById(hourly_meanHeartrate_by_Id, meanHeartrate_by_hour, mean_heartrate, slice_tuple, title):
    Id_list = list(hourly_meanHeartrate_by_Id['Id'].unique())
    hourly_heartrate_by_id = {}
    for Id in Id_list:
        heartrates = {}
        sub_df = hourly_meanHeartrate_by_Id[hourly_meanHeartrate_by_Id['Id'] == Id]
        for row in zip(sub_df['hour'], sub_df['meanHeartrate']):
            heartrates[row[0]] = row[1]
        hourly_heartrate_by_id[Id] = heartrates 
        
    # plot portion of the Id_list
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.xticks(range(0, 24))
    for item in Id_list[slice_tuple[0]:slice_tuple[1]]:
        xs = list(hourly_heartrate_by_id[item].keys())
        ys = list(hourly_heartrate_by_id[item].values())
        rgb = f"#{secrets.token_hex(3)}"
        ax.plot(xs, ys, label=item, color=rgb, marker='x')
    sns.scatterplot(data=meanHeartrate_by_hour, x='hour', y='meanHeartrate', label='Overall', color='black')
    sns.lineplot(data=meanHeartrate_by_hour, x='hour', y='meanHeartrate', label='Overall', color='black', ls='--')
    plt.axhline(mean_heartrate, ls='--', color='black')
    plt.figtext(0.8, 0.06, f'Overall Mean Heartrate: {mean_heartrate}', ha="center", fontsize=11)
    plt.title(title)
    plt.xlabel('Hour')
    plt.ylabel('Heartrate')
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.show()
    
def plot_hourly_caloriesById(hourlyCalories_by_Id, hourlyCalories, mean_calories, slice_tuple, title):
    Id_list = list(hourlyCalories_by_Id['Id'].unique())
    hourlyCalories_by_id = {}
    for Id in Id_list:
        calories = {}
        sub_df = hourlyCalories_by_Id[hourlyCalories_by_Id['Id'] == Id]
        for row in zip(sub_df['hour'], sub_df['meanCalories']):
            calories[row[0]] = row[1]
        hourlyCalories_by_id[Id] = calories
        
    # plot portion of the Id_list  
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.xticks(range(0, 24))
    for item in Id_list[slice_tuple[0]:slice_tuple[1]]:
        xs = list(hourlyCalories_by_id[item].keys())
        ys = list(hourlyCalories_by_id[item].values())
        rgb = f"#{secrets.token_hex(3)}"
        ax.plot(xs, ys, label=item, color=rgb, marker='x')
    sns.scatterplot(data=hourlyCalories, x='hour', y='meanCalories', label='Overall', color='black')
    sns.lineplot(data=hourlyCalories, x='hour', y='meanCalories', label='Overall', color='black', ls='--')
    # plot mean calaries with black horizontal dotted line
    plt.axhline(mean_calories, ls='--', color='black')
    plt.figtext(0.8, 0.06, f'Overall Mean Calories: {mean_calories}', ha="center", fontsize=11)
    plt.title(title)
    plt.xlabel('Hour')
    plt.ylabel('Calories')
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.show()
    

# Processing functions #
def generate_mean_weights(df):
    avg_weights = []
    for coef in ['coef1', 'coef2', 'coef3']:
        trials =df[coef].to_list()
        avg_weights.append(round(sum(trials) / len(trials), 2))
    return avg_weights

def add_weightedActivity_columns(df, weights):
    df = df.copy()
    df['weightedVeryActiveMinutes'] = df['VeryActiveMinutes'].apply(lambda x: x*weights[0])
    df['weightedFairlyActiveMinutes'] = df['FairlyActiveMinutes'].apply(lambda x: x*weights[1])
    df['weightedLightlyActiveMinutes'] = df['LightlyActiveMinutes'].apply(lambda x: x*weights[2])
    df['weightedModerateActivity_combined'] = df[['weightedVeryActiveMinutes', 'weightedFairlyActiveMinutes']].apply(lambda x: (x.values.sum()), axis="columns")
    df['weightedActivity_combined'] = df[['weightedVeryActiveMinutes', 'weightedFairlyActiveMinutes', 'weightedLightlyActiveMinutes']].apply(lambda x: (x.values.sum()), axis="columns")
    return df

def generate_activity_minutes_weights(df, x_cols, y_col):
    y = df[y_col]
    x = df[x_cols]
    x=x.to_numpy()
    
    # create a list of 3 random states to loop through
    
    random_states = [23, 7, 45]
    trial_results = []
    
    for i, item in enumerate(random_states):
        # splitting the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=item)
        # creating an object of LinearRegression class
        LR = LinearRegression()
        # fitting the training data
        LR.fit(x_train,y_train)
        
        coef_df = pd.DataFrame(LR.coef_.T, columns=['Coefficient'])
        activityCoefs = coef_df['Coefficient'].to_list()
        y_prediction =  LR.predict(x_test)
        # predicting the accuracy score
        score=r2_score(y_test,y_prediction)
        trial_results_cols = ['coef1', 'coef2', 'coef3', 'r2_score', 'mean_squared_error', 'root_mean_squared_error']
        trial_results.append([activityCoefs[0], activityCoefs[1], activityCoefs[2], score, mean_squared_error(y_test,y_prediction), np.sqrt(mean_squared_error(y_test,y_prediction))])
        trial_results_df = pd.DataFrame(trial_results, columns=trial_results_cols)
        
        print(f'Trial {i+1}')
        print('r2 score is:',score)
        print('mean_sqrd_error is:',mean_squared_error(y_test,y_prediction))
        print('root_mean_squared_error of is:', np.sqrt(mean_squared_error(y_test,y_prediction)))
        
    trial_results_df = pd.DataFrame(trial_results, columns=trial_results_cols)
    return trial_results_df

def createAndPopulate_database(file_path):
    files = os.listdir(file_path)
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(f'{file_path}{file}')
            table_name = file[:-4]
            query_string = f'CREATE TABLE IF NOT EXISTS "{table_name}" ('
            for col_name, dtype in zip(df.columns, df.dtypes):
                if dtype == 'int64':
                    sql_dtype = 'INTEGER'
                elif dtype == 'float64':
                    sql_dtype = 'REAL'
                elif dtype == 'object':
                    sql_dtype = 'TEXT'
                else:
                    print('PROBLEM')
                query_string += f'"{col_name}" {sql_dtype}, '
            query_string = query_string[:-2]
            query_string += ');'
            sql_execute(query_string)
        
    print('SQL Database created and populated')
    
def populateTables(file_path, verbose=False):
    connection = sqlite3.connect('fitbit_database.db')
    cursor = connection.cursor()
    files = os.listdir(file_path)
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(f'{file_path}{file}')
            table_name = file[:-4]
            df_cols = list(df.columns)
            for row in df.iterrows():
                query_string = f'INSERT OR IGNORE INTO {table_name} ('
                row_values = []
                for i, col in enumerate(df_cols):
                    query_string += f'{col}, '
                    row_values.append(row[1][i])

                query_string = query_string[:-2]
                query_string += ') VALUES ('
                query_string += '?, ' * len(df_cols)
                query_string = query_string[:-2]
                row_values_tuple = tuple(i for i in row_values)
                query_string += ')'
                cursor.execute(query_string, row_values_tuple)
                
            connection.commit()
            if verbose:
                print(f'{table_name} populated')
    cursor.close()

# Processing functions #
def make_dt_columns_second(df, datetime_column, value_columns):
    # make value_columns param a list of interested columns
    df['dtString'] = df[datetime_column].astype('datetime64[ns]')
    df = df.drop(columns=datetime_column)
    df['month'] = df['dtString'].dt.month
    df['day'] = df['dtString'].dt.day
    df['wkDay'] = df['dtString'].dt.weekday
    df['hour'] = df['dtString'].dt.hour
    df['minute'] = df['dtString'].dt.minute
    df['second'] = df['dtString'].dt.second
    columns_order = ['Id', 'dtString', 'month', 'day', 'wkDay', 'hour', 'minute', 'second']
    if len(value_columns) == 1:
        columns_order.append(value_columns[0])
    elif len(value_columns) > 1:
        for item in value_columns:
            columns_order.append(item)
    df = df[columns_order]
    return df

def make_dt_columns_minute(df, datetime_column, value_columns):
    # make value_columns param a list of interested columns
    df['dtString'] = df[datetime_column].astype('datetime64[ns]')
    df = df.drop(columns=datetime_column)
    df['month'] = df['dtString'].dt.month
    df['day'] = df['dtString'].dt.day
    df['wkDay'] = df['dtString'].dt.weekday
    df['hour'] = df['dtString'].dt.hour
    df['minute'] = df['dtString'].dt.minute
    columns_order = ['Id', 'dtString', 'month', 'day', 'wkDay', 'hour', 'minute']
    if len(value_columns) == 1:
        columns_order.append(value_columns[0])
    elif len(value_columns) > 1:
        for item in value_columns:
            columns_order.append(item)
    df = df[columns_order]
    return df

def make_dt_columns_hourly(df, datetime_column, value_columns):
    # make value_columns param a list of interested columns
    df['dtString'] = df[datetime_column].astype('datetime64[ns]')
    df = df.drop(columns=datetime_column)
    df['month'] = df['dtString'].dt.month
    df['day'] = df['dtString'].dt.day
    df['wkDay'] = df['dtString'].dt.weekday
    df['hour'] = df['dtString'].dt.hour
    columns_order = ['Id', 'dtString', 'month', 'day', 'wkDay', 'hour']
    if len(value_columns) == 1:
        columns_order.append(value_columns[0])
    elif len(value_columns) > 1:
        for item in value_columns:
            columns_order.append(item)
    df = df[columns_order]
    return df

def make_dt_columns_daily(df, datetime_column):
    # make value_columns param a list of interested columns
    df['dtString'] = df[datetime_column].astype('datetime64[ns]')
    df = df.drop(columns=datetime_column)
    df['month'] = df['dtString'].dt.month
    df['day'] = df['dtString'].dt.day
    df['wkDay'] = df['dtString'].dt.weekday
    columns_order = ['Id', 'dtString', 'month', 'day', 'wkDay']
    value_cols = df.drop(columns_order, axis=1).columns
    if len(value_cols) == 1:
        columns_order.append(value_cols[0])
    elif len(value_cols) > 1:
        for item in value_cols:
            columns_order.append(item)
    df = df[columns_order]
    return df

def loadAndProcess(file_name, time_interval, datetime_column, value_columns, cleaned=False):
    # make value_columns param a list of interested columns
    if cleaned:
        csv_file_str = f'data/{file_name}_cleaned_merged.csv'
    else:
        csv_file_str = f'data/{file_name}_merged.csv'
    df_raw = pd.read_csv(csv_file_str)
    if time_interval =='second':
        df = make_dt_columns_second(df_raw, datetime_column, value_columns)
    if time_interval =='minute':
        df = make_dt_columns_minute(df_raw, datetime_column, value_columns)
    elif time_interval == 'hour':
        df = make_dt_columns_hourly(df_raw, datetime_column, value_columns)
    elif time_interval == 'day':
        df = make_dt_columns_daily(df_raw, datetime_column)
    output_file_str = f'data_processed/{file_name}_processed.csv'
    df.to_csv(output_file_str, index=False)
    print(f'{file_name} processed')
    
# generate #
def loadAndGenerate_minuteHeartrate():
    heartrateSeconds = pd.read_csv('data_processed/heartrate_seconds_processed.csv')
    minuteHeartrate = heartrateSeconds.groupby(['Id', 'month', 'day', 'hour', 'minute']).agg({'Value':['min', 'max', 'mean']}).reset_index()
    minuteHeartrate.columns = minuteHeartrate.columns.droplevel(1)
    minuteHeartrate.columns = ['Id', 'month', 'day', 'hour', 'minute', 'min', 'max', 'mean']
    cols1 = ['month', 'day']
    cols2 = ['dtString', 'hour']
    minuteHeartrate['minute'] = minuteHeartrate['minute'].astype(str).apply(lambda x: x.zfill(2))
    cols3 = ['dtString', 'minute']
    minuteHeartrate['dtString'] = minuteHeartrate[cols1].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    minuteHeartrate['dtString'] = minuteHeartrate['dtString'].apply(lambda x: f'{x}-2016')
    minuteHeartrate['dtString'] = minuteHeartrate[cols2].apply(lambda x: ' '.join(x.values.astype(str)), axis="columns")
    minuteHeartrate['dtString'] = minuteHeartrate[cols3].apply(lambda x: ':'.join(x.values.astype(str)), axis="columns")
    minuteHeartrate['dtString'] = minuteHeartrate['dtString'].apply(lambda x: f'{x.zfill(2)}:00')
    minuteHeartrate['dtString'] = pd.to_datetime(minuteHeartrate['dtString'])
    minuteHeartrate['minute'] = minuteHeartrate['minute'].astype('int64')
    minuteHeartrate['wkDay'] = minuteHeartrate['dtString'].dt.weekday
    minuteHeartrate = minuteHeartrate[['Id', 'dtString', 'month', 'day', 'wkDay', 'hour', 'minute', 'min', 'max', 'mean']]
    minuteHeartrate.to_csv('data_processed/minuteHeartrate_generated.csv', index=False)
    print('minuteHeartrate generated')
    
def loadAndGenerate_hourlyHeartrate():
    heartrateSeconds = pd.read_csv('data_processed/heartrate_seconds_processed.csv')
    hourHeartrate = heartrateSeconds.groupby(['Id', 'month', 'day', 'hour']).agg({'Value':['min', 'max', 'mean']}).reset_index()
    hourHeartrate.columns = hourHeartrate.columns.droplevel(1)
    hourHeartrate.columns = ['Id', 'month', 'day', 'hour', 'min', 'max', 'mean']
    cols1 = ['month', 'day']
    cols2 = ['dtString', 'hour']
    hourHeartrate['dtString'] = hourHeartrate[cols1].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    hourHeartrate['dtString'] = hourHeartrate['dtString'].apply(lambda x: f'{x}-2016')
    hourHeartrate['dtString'] = hourHeartrate[cols2].apply(lambda x: ' '.join(x.values.astype(str)), axis="columns")
    hourHeartrate['dtString'] = hourHeartrate['dtString'].apply(lambda x: f'{x.zfill(2)}:00:00')
    hourHeartrate['dtString'] = pd.to_datetime(hourHeartrate['dtString'])
    hourHeartrate['wkDay'] = hourHeartrate['dtString'].dt.weekday
    hourHeartrate = hourHeartrate[['Id', 'dtString', 'month', 'day', 'wkDay', 'hour', 'min', 'max', 'mean']]
    hourHeartrate.to_csv('data_processed/hourlyHeartrate_generated.csv', index=False)
    print('hourlyHeartrate generated')
    
def loadAndGenerate_hourlyMETs():
    minuteMETsNarrow = pd.read_csv('data_processed/minuteMETsNarrow_processed.csv')
    hourlyMETsNarrow = minuteMETsNarrow.groupby(['Id', 'month', 'day', 'hour']).agg({'METs': ['sum', 'mean']}).reset_index()
    hourlyMETsNarrow.columns = hourlyMETsNarrow.columns.droplevel(1)
    hourlyMETsNarrow.columns = ['Id', 'month', 'day', 'hour', 'METs', 'avgMETs']
    cols1 = ['month', 'day']
    cols2 = ['dtString', 'hour']
    hourlyMETsNarrow['dtString'] = hourlyMETsNarrow[cols1].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    hourlyMETsNarrow['dtString'] = hourlyMETsNarrow['dtString'].apply(lambda x: f'{x}-2016')
    hourlyMETsNarrow['dtString'] = hourlyMETsNarrow[cols2].apply(lambda x: ' '.join(x.values.astype(str)), axis="columns")
    hourlyMETsNarrow['dtString'] = hourlyMETsNarrow['dtString'].apply(lambda x: f'{x.zfill(2)}:00:00')
    hourlyMETsNarrow['dtString'] = pd.to_datetime(hourlyMETsNarrow['dtString'])
    hourlyMETsNarrow['wkDay'] = hourlyMETsNarrow['dtString'].dt.weekday
    hourlyMETsNarrow = hourlyMETsNarrow[['Id', 'dtString', 'month', 'day', 'wkDay', 'hour', 'METs', 'avgMETs']]
    hourlyMETsNarrow.to_csv('data_processed/hourlyMETs_generated.csv', index=False)
    print('hourlyMETs generated')
    
def loadAndGenerate_dailyMETS():
    minuteMETsNarrow = pd.read_csv('data_processed/minuteMETsNarrow_processed.csv')
    dailyMETsNarrow = minuteMETsNarrow.groupby(['Id', 'month', 'day']).agg({'METs': ['sum', 'mean']}).reset_index()
    dailyMETsNarrow.columns = dailyMETsNarrow.columns.droplevel(1)
    dailyMETsNarrow.columns = ['Id', 'month', 'day', 'totalMETs', 'avgMETs']
    cols1 = ['month', 'day']
    dailyMETsNarrow['dtString'] = dailyMETsNarrow[cols1].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    dailyMETsNarrow['dtString'] = dailyMETsNarrow['dtString'].apply(lambda x: f'{x}-2016')
    dailyMETsNarrow['dtString'] = pd.to_datetime(dailyMETsNarrow['dtString'])
    dailyMETsNarrow['wkDay'] = dailyMETsNarrow['dtString'].dt.weekday
    dailyMETsNarrow = dailyMETsNarrow[['Id', 'dtString', 'month', 'day', 'wkDay', 'totalMETs', 'avgMETs']]
    dailyMETsNarrow.to_csv('data_processed/dailyMETsNarrow_generated.csv', index=False)
    print('dailyMETs generated')
    
def loadAndGenerate_hourlySleep():
    minuteSleep = pd.read_csv(r'data_processed/minuteSleep_processed.csv')
    hourlySleep = minuteSleep.groupby(['Id', 'month', 'day', 'hour']).agg({'value': ['count', 'sum', 'mean'], 'logId': ['mean']}).reset_index()
    hourlySleep.columns = hourlySleep.columns.droplevel(1)
    hourlySleep.columns = ['Id', 'month', 'day', 'hour', 'totalSleepMinute', 'totalSleepQuality', 'avgSleepQuality', 'avglogId']
    cols1 = ['month', 'day']
    cols2 = ['dtString', 'hour']
    hourlySleep['dtString'] = hourlySleep[cols1].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    hourlySleep['dtString'] = hourlySleep['dtString'].apply(lambda x: f'{x}-2016')
    hourlySleep['dtString'] = hourlySleep[cols2].apply(lambda x: ' '.join(x.values.astype(str)), axis="columns")
    hourlySleep['dtString'] = hourlySleep['dtString'].apply(lambda x: f'{x.zfill(2)}:00:00')
    hourlySleep['dtString'] = pd.to_datetime(hourlySleep['dtString'])
    hourlySleep['wkDay'] = hourlySleep['dtString'].dt.weekday
    hourlySleep = hourlySleep[['Id', 'dtString', 'month', 'day', 'wkDay', 'hour', 'totalSleepMinute', 'totalSleepQuality', 'avgSleepQuality', 'avglogId']]
    hourlySleep.to_csv('data_processed/hourlySleep_generated.csv', index=False)
    print('hourlySleep generated')
    
def loadAndGenerate_dailySleep():
    minuteSleep = pd.read_csv(r'data_processed/minuteSleep_processed.csv')
    dailySleep = minuteSleep.groupby(['Id', 'month', 'day']).agg({'value': ['count', 'sum', 'mean'], 'logId': ['mean']}).reset_index()
    dailySleep.columns = dailySleep.columns.droplevel(1)
    dailySleep.columns = ['Id', 'month', 'day', 'totalSleepMinute', 'totalSleepQuality', 'avgSleepQuality', 'avglogId']
    cols1 = ['month', 'day']
    dailySleep['dtString'] = dailySleep[cols1].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    dailySleep['dtString'] = dailySleep['dtString'].apply(lambda x: f'{x}-2016')
    dailySleep['dtString'] = pd.to_datetime(dailySleep['dtString'])
    dailySleep['wkDay'] = dailySleep['dtString'].dt.weekday
    dailySleep = dailySleep[['Id', 'dtString', 'month', 'day', 'wkDay', 'totalSleepMinute', 'totalSleepQuality', 'avgSleepQuality', 'avglogId']]
    dailySleep.to_csv('data_processed/dailySleep_generated.csv', index=False)
    print('dailySleep generated')









