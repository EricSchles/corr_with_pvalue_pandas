def correlation_function(func_name, first, second):
    if func_name == "pearson":
        return stats.pearsonr(first, second)
    elif func_name == "spearman":
        return stats.spearmanr(first, second)
    elif func_name == "point biserial":
        return stats.pointbiserialr(first, second)

    
def corr(df, how="pearson"):
    tmp = df.copy()
    for column in tmp.columns:
        if tmp[column].dtype != np.float64:
            tmp.drop(column, axis=1, inplace=True)
    df1 = tmp.copy()
    df2 = tmp.copy()
    correlate = partial(correlation_function, how)
    coef_matrix = np.zeros((df1.shape[1], df2.shape[1]))
    pval_matrix = np.zeros((df1.shape[1], df2.shape[1]))

    for index_one in range(df1.shape[1]):
        for index_two in range(df2.shape[1]):
            corr_test = correlate(df1[df1.columns[index_one]], df2[df2.columns[index_two]])
            coef_matrix[index_one, index_two] = corr_test[0]
            pval_matrix[index_one, index_two] = corr_test[1]
    df_coef = pd.DataFrame(coef_matrix, columns=df2.columns, index=df1.columns)
    df_pval = pd.DataFrame(pval_matrix, columns=df2.columns, index=df1.columns)
    return df_coef, df_pval
