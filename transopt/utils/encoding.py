import pandas as pds

def target_encoding(df:pds.DataFrame, column_name, target_name):
    """
    计算给定列的目标编码。

    参数:
    dataframe (pandas.DataFrame): 包含特征和目标列的DataFrame。
    column_name (str): 需要进行目标编码的列名。
    target_name (str): 目标变量的列名。

    返回:
    dict: 包含每个唯一值及其目标编码的字典。
    """
    # 计算每个唯一值的目标均值
    target_mean = df.groupby(column_name)[target_name].mean()

    # 返回结果字典
    return target_mean.to_dict()

def multitarget_encoding(df:pds.DataFrame, column_name, target_names):
    encodings = {}
    for target in target_names:
        # 计算每个唯一值的目标均值
        target_mean = df.groupby(column_name)[target].mean()
        encodings[target] = target_mean.to_dict()
    return encodings