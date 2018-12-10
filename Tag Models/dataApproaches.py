##Data Approaches


####a_tMax and t_tMax
analyst_data = pd.read_csv(data_directory+"analystTopic.csv")
a_topic_cols = analyst_data.drop(['AnalystName', 'EventNumber', 'a_tMax'], axis=1).columns.tolist()
topic_cols = a_topic_cols.copy()
indexed_data = indexed_data.merge(analyst_data[['AnalystName', 'EventNumber', 'a_tMax']], on=['AnalystName','EventNumber'])

tag_data = pd.read_csv(data_directory+"tagTopic.csv")
t_topic_cols = tag_data.drop(['Tag', 'EventNumber', 't_tMax'], axis=1).columns.tolist()
topic_cols += t_topic_cols.copy()
indexed_data = indexed_data.merge(tag_data[['Tag', 'EventNumber', 't_tMax']], on=['Tag','EventNumber'])

cols_save = ['EventNumber','Company', 'Month', 'Year', 'Quarter', 'EventType', 'Date']

pivot_data = pd.pivot_table(indexed_data, index=cols_save, columns=['a_tMax','t_tMax'], aggfunc='size', fill_value=0).reset_index()
pivot_data.columns = ['__'.join(col).rstrip('__') for col in pivot_data.columns.values]
melt_data = pd.melt(pivot_data, id_vars=cols_save, var_name=['a_tMax__t_tMax'], value_name='NumQ')
melt_data[['a_tMax', 't_tMax']] = melt_data['a_tMax__t_tMax'].str.split("__", expand=True)
melt_data['NumQ'] = melt_data['NumQ'].astype(bool).astype(int)
melt_data = pd.concat([melt_data,
                          pd.get_dummies(melt_data[['a_tMax', 't_tMax', 'Company', 'EventType']])], axis=1).reset_index(drop=True)

features_data = melt_data.drop(['Company', 'EventType', 'a_tMax', 't_tMax', 'a_tMax__t_tMax', 'Date'], axis=1).copy()
train, test = features_data.loc[~features_data['EventNumber'].isin(test_set)].copy().reset_index(drop=True), \
                features_data.loc[features_data['EventNumber'].isin(test_set)].copy().reset_index(drop=True)

X_train, y_train = train.drop(['NumQ','EventNumber'], axis=1).values, train['NumQ'].values
X_test, y_test = test.drop(['NumQ', 'EventNumber'], axis=1).values, test['NumQ'].values

cols_list = train.drop(['NumQ','EventNumber'], axis=1).columns.values

####a_t and t_t weight and a_tMax and t_tMax
analyst_data = pd.read_csv(data_directory+"analystTopic.csv")
a_topic_cols = analyst_data.drop(['AnalystName', 'EventNumber', 'a_tMax'], axis=1).columns.tolist()
topic_cols = a_topic_cols.copy()
indexed_data = indexed_data.merge(analyst_data, on=['AnalystName','EventNumber'])

tag_data = pd.read_csv(data_directory+"tagTopic.csv")
t_topic_cols = tag_data.drop(['Tag', 'EventNumber', 't_tMax'], axis=1).columns.tolist()
topic_cols += t_topic_cols.copy()
indexed_data = indexed_data.merge(tag_data, on=['Tag','EventNumber'])

cols_save = ['EventNumber','Company', 'Month', 'Year', 'Quarter', 'EventType', 'Date'] + a_topic_cols + t_topic_cols

pivot_data = pd.pivot_table(indexed_data, index=cols_save, columns=['a_tMax','t_tMax'], aggfunc='size', fill_value=0).reset_index()
pivot_data.columns = ['__'.join(col).rstrip('__') for col in pivot_data.columns.values]

melt_data = pd.melt(pivot_data, id_vars=cols_save, var_name=['a_tMax__t_tMax'], value_name='NumQ')
melt_data[['a_tMax', 't_tMax']] = melt_data['a_tMax__t_tMax'].str.split("__", expand=True)
melt_data['NumQ'] = melt_data['NumQ'].astype(bool).astype(int)
melt_data = pd.concat([melt_data,
                      pd.get_dummies(melt_data[['a_tMax', 't_tMax']])], axis=1).reset_index(drop=True)

features_data = melt_data.drop(['Company', 'EventType', 't_tMax', 'Date', 'a_tMax'  , 'a_tMax__t_tMax'], axis=1).copy()
train, test = features_data.loc[~features_data['EventNumber'].isin(test_set)].copy().reset_index(drop=True), \
                features_data.loc[features_data['EventNumber'].isin(test_set)].copy().reset_index(drop=True)

X_train, y_train = train.drop(['NumQ','EventNumber'], axis=1).values, train['NumQ'].values
X_test, y_test = test.drop(['NumQ', 'EventNumber'], axis=1).values, test['NumQ'].values

cols_list = train.drop(['NumQ','EventNumber'], axis=1).columns.values

##Final a_tMax, t_tMax
analyst_data = pd.read_csv(data_directory+"analystTopic.csv")
analyst_tail = analyst_data.groupby(['AnalystName']).tail(n=1).drop(['EventNumber'], axis=1)
indexed_data = indexed_data.merge(analyst_tail[['AnalystName', 'a_tMax']], on=['AnalystName'])

tag_data = pd.read_csv(data_directory+"tagTopic.csv")
tag_tail = tag_data.groupby(['Tag']).tail(n=1).drop(['EventNumber'], axis=1)
indexed_data = indexed_data.merge(tag_tail[['Tag', 't_tMax']], on=['Tag'])

a_topic_cols = analyst_tail.drop(['AnalystName', 'a_tMax'], axis=1).columns.tolist()
topic_cols = a_topic_cols.copy()
t_topic_cols = tag_tail.drop(['Tag', 't_tMax'], axis=1).columns.tolist()
topic_cols += t_topic_cols.copy()

cols_save = ['EventNumber','Company', 'Month', 'Year', 'Quarter', 'EventType', 'Date']

pivot_data = pd.pivot_table(indexed_data, index=cols_save, columns=['a_tMax','t_tMax'], aggfunc='size', fill_value=0).reset_index()
pivot_data.columns = ['__'.join(col).rstrip('__') for col in pivot_data.columns.values]
melt_data = pd.melt(pivot_data, id_vars=cols_save, var_name=['a_tMax__t_tMax'], value_name='NumQ')
melt_data[['a_tMax', 't_tMax']] = melt_data['a_tMax__t_tMax'].str.split("__", expand=True)
melt_data['NumQ'] = melt_data['NumQ'].astype(bool).astype(int)
melt_data = pd.concat([melt_data,
                          pd.get_dummies(melt_data[['a_tMax', 't_tMax', 'Company', 'EventType']])], axis=1).reset_index(drop=True)

features_data = melt_data.drop(['Company', 'EventType', 'a_tMax', 't_tMax', 'a_tMax__t_tMax', 'Date'], axis=1).copy()
train, test = features_data.loc[~features_data['EventNumber'].isin(test_set)].copy().reset_index(drop=True), \
                features_data.loc[features_data['EventNumber'].isin(test_set)].copy().reset_index(drop=True)

X_train, y_train = train.drop(['NumQ','EventNumber'], axis=1).values, train['NumQ'].values
X_test, y_test = test.drop(['NumQ', 'EventNumber'], axis=1).values, test['NumQ'].values

cols_list = train.drop(['NumQ','EventNumber'], axis=1).columns.values
