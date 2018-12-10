def historicVarPct(data, var_indep, var_dep):
    
    data['CompanyDate'] = data['Company'] + "___" + data['Date'].astype(str)
    
    var_dep_cols = data[var_dep].unique().tolist()
    overall_hist_pct_list = [0 for _ in range(data[var_indep].unique().shape[0])]
    hist_dep_cols = ['{}Hist_{}'.format(var_indep, t) for t in var_dep_cols]

    for c, (i, a_group) in enumerate(data.groupby([var_indep])):

        subgroups = [(d, g) for d, g in a_group.groupby(['CompanyDate'])]
        hist_pct_list = [pd.DataFrame(columns=[var_indep, 'CompanyDate'] + hist_dep_cols) for _ in range(len(subgroups))]

        for j, (d,g) in enumerate(subgroups):
            if j == 0:
                hist_pct_list[j] = hist_pct_list[j].append(pd.DataFrame({var_indep:[i], 'CompanyDate':[subgroups[j][0]]}), sort=False).fillna(0)
                continue

            hist_data = pd.concat([s_g[1] for s_g in subgroups[:j]]).groupby([var_dep], sort='True').size().reset_index()
            hist_data = hist_data.append(pd.DataFrame([(t, 0.0) for t in np.setdiff1d(var_dep_cols, hist_data[var_dep].values)], columns=[var_dep, 0]), sort=False)
            hist_data[var_dep+'Pct'] = hist_data[0]/hist_data[0].sum()
            hist_data['CompanyDate'] = d

            hist_pivot = hist_data.pivot(index='CompanyDate', columns=var_dep, values=var_dep+"Pct").fillna(0).reset_index()
            hist_pivot.columns = ['CompanyDate'] + ["{}Hist_{}".format(var_indep, c) for c in hist_pivot.columns[1:]]
            hist_pivot.index = [i]
            hist_pivot = hist_pivot.rename_axis(None, axis=1).rename_axis(var_indep).reset_index()
            hist_pct_list[j] = hist_pct_list[j].append(hist_pivot, sort=False).fillna(0)

        overall_hist_pct_list[c] = pd.concat(hist_pct_list)

    overall_hist_pct = pd.concat(overall_hist_pct_list)
    overall_hist_pct[['Company', 'Date']] = overall_hist_pct['CompanyDate'].str.split("___", expand=True)
    overall_hist_pct['Date'] = pd.to_datetime(overall_hist_pct['Date'])
    overall_hist_pct.drop(['CompanyDate'], axis=1, inplace=True)
    
    data.drop(["CompanyDate"], axis=1, inplace=True)
    
    return(overall_hist_pct)