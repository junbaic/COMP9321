import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math
import re

studentid = os.path.basename(sys.modules[__name__].__file__)


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())


def func1(item):
    item = str(item)
    while item.find('(') >= 0:
        index1 = item.find('(')
        index2 = item.find(')', index1)
        item = item[:index1] + item[index2 + 1:]
    while item.find('via') >= 0:
        index1 = item.find('via')
        item = item[:index1]
    item = item.replace(' to ', ',')
    item = item.replace(' and ', ',')
    item = item.replace(' then ', ',')
    splitted = item.strip().split(',')
    first = splitted[0].strip()
    end = splitted[-1].strip()
    return first


def func2(item):
    item = str(item)
    while item.find('(') >= 0:
        index1 = item.find('(')
        index2 = item.find(')', index1)
        item = item[:index1] + item[index2 + 1:]
    while item.find('via') >= 0:
        index1 = item.find('via')
        item = item[:index1]
    item = item.replace(' to ', ',')
    item = item.replace(' and ', ',')
    item = item.replace(' then ', ',')
    splitted = item.strip().split(',')
    first = splitted[0].strip()
    end = splitted[-1].strip()
    return end


def question_1(routes, suburbs):
    """
    :param routes: the path for the routes dataset
    :param suburbs: the path for the routes suburbs
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # Your code goes here ...
    df1 = pd.read_csv(routes)
    start = df1['service_direction_name'].apply(func1)
    end = df1['service_direction_name'].apply(func2)
    df1['start'] = start
    df1['end'] = end
    #################################################

    log("QUESTION 1", output_df=df1[["service_direction_name", "start", "end"]], other=df1.shape)
    return df1


def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: dataframe df2
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    total_list = df1['start'].values.tolist() + df1['end'].values.tolist()
    df = pd.DataFrame({"A": total_list})
    df2 = pd.DataFrame(df['A'].value_counts(ascending=False))
    df2['index'] = df2.index
    df2_ = df2['index']
    df2 = df2.drop('index', axis=1)
    df2.insert(0, 'service_location', df2_)
    df2.columns = ['service_location', 'frequency']
    df2 = df2.reset_index()
    df2 = df2.drop('index', axis=1)
    #################################################
    log("QUESTION 2", output_df=df2, other=df2.shape)
    return df2


def question_3(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # Your code goes here ...
    df3 = df1.copy(True)
    df3['transport_name'] = df3['transport_name'].str.lower()
    df3['transport_name'] = df3['transport_name'].apply(
        lambda x: 'Bus' if 'bus' in x else x)
    df3['transport_name'] = df3['transport_name'].apply(
        lambda x: 'Ferr' if 'Ferry' in x else x)
    df3['transport_name'] = df3['transport_name'].apply(
        lambda x: 'Light Rail' if 'Light Rail' in x else x)
    df3['transport_name'] = df3['transport_name'].apply(
        lambda x: 'Trains' if 'Train' in x else x)
    df3['transport_name'] = df3['transport_name'].apply(
        lambda x: 'Metro' if 'Metro' in x else x)
    #################################################

    log("QUESTION 3", output_df=df3[['transport_name']], other=df3.shape)
    return df3


def question_4(df3):
    """
    :param df3: the dataframe created in question 3
    :param continents: the path for the Countries-Continents.csv file
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    temp = df3['transport_name'].value_counts().to_frame()
    temp_t = temp.T
    trans = list(temp_t.columns)
    trans_ = trans[::-1]
    fre = (temp_t.values.flatten())
    fre_ = fre[::-1]
    df4 = pd.DataFrame({'transport_name': trans_, 'frequency': fre_})
    #################################################

    log("QUESTION 4", output_df=df4[["transport_name", "frequency"]], other=df4.shape)
    return df4


def question_5(df3, suburbs):
    """
    :param df3: the dataframe created in question 2
    :param suburbs : the path to dataset
    :return: df5
            Data Type: dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # Your code goes here ...
    suburbs_df = pd.read_csv(suburbs)
    df_a = suburbs_df[['suburb', 'population']]
    df_b = df3[df3['depot_name'].notnull()]['depot_name'].value_counts().to_frame()
    df_b['frequency'] = df_b['depot_name']
    df_b['depot_name'] = df_b.index
    dfa = df_a.merge(right=df_b, how='inner', left_on="suburb", right_on="depot_name")
    dfa = dfa.sort_values(by='frequency', ascending=False)[:5]
    df5 = pd.DataFrame()
    df5['ratio'] = dfa['population']
    df5['depot'] = dfa['depot_name']
    df5 = df5.set_index('depot')
    #################################################

    log("QUESTION 5", output_df=df5[["ratio"]], other=df5.shape)
    return df5


def question_6(df3):
    """
    :param df3: the dataframe created in question 3
    :return: pandas pivot table
    """
    table = None
    #################################################
    # Your code goes here ...
    table = pd.DataFrame()
    df4 = df3[['operator_name', 'transport_name']]
    table['operator_name'] = df4.groupby(by='operator_name').groups
    table.set_index('operator_name')
    for s in df3.transport_name:
        table[s] = 0
    g = df3.groupby(by='transport_name')
    for s, k in g:
        for i, m in (k['operator_name'].value_counts()).items():
            table.loc[i, s] = m
    table = table.drop(columns='operator_name')
    table['sum_of_all_routes'] = table.iloc[:, :-1].sum(axis=1)
    table['sum_of_all_transport'] = table[table > 0].notnull().sum(axis=1)
    #################################################

    log("QUESTION 6", output_df=None, other=table)
    return table


def question_7(df3, suburbs):
    """
    :param df3: the dataframe created in question 3
    :param suburbs : the path to dataset
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    suburbs_df = pd.read_csv(suburbs)
    df7 = suburbs_df[suburbs_df['statistic_area'].str.contains('Greater Sydney')]
    df7 = df7[
        ((df7['population'] > 0) & (df7['median_income'] > 0)) | (
                (df7['population'] == 0) & (df7['median_income'] == 0))]
    gb = df7.groupby(by='local_goverment_area')
    key = ['population', 'median_income', 'sqkm']
    data = {k: [] for k in key}
    for l, group in gb:
        for k in key:
            data[k].append(group[k].sum())
    data['l'] = list(gb.groups)
    plt.style.use('seaborn')
    size = np.array(data['population'])/500
    plt.scatter(data['sqkm'], data['median_income'], s=size, c=np.arange(len(data['l'])), cmap='flag')
    plt.tight_layout()
    plt.xlabel('LGA')
    plt.ylabel('median_sum')
    plt.show()
    #################################################

    plt.savefig("{}-Q7.png".format(studentid))


def question_8(df3, suburbs):
    """
    :param df3: the dataframe created in question 3
    :param suburbs : the path to dataset
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    suburbs_df = pd.read_csv(suburbs)
    df8 = suburbs_df[suburbs_df['state'] == 'NSW']
    lat = df8['lat']
    lng = df8['lng']
    plate = df8['sqkm']
    df8 = df3
    plt.hist2d(lng, lat, bins=400, weights=plate)
    plt.clim(0, 5)
    plt.xlabel('lng')
    plt.ylabel('lat')
    plt.show()
    #################################################

    plt.savefig("{}-Q8.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("routes.csv", "suburbs.csv")
    df2 = question_2(df1.copy(True))
    df3 = question_3(df1.copy(True))
    df4 = question_4(df3.copy(True))
    df5 = question_5(df3.copy(True), "suburbs.csv")
    table = question_6(df3.copy(True))
    question_7(df3.copy(True), "suburbs.csv")
    question_8(df3.copy(True), "suburbs.csv")
