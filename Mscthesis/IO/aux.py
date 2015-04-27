

def built_network_from_neighs(df, type_var, neighs_dir):
    """Function for building the network from neighbours."""

    type_vals = list(df[type_var].unique())
    n_vals = len(type_vals)

    N_t = df.shape[0]
    N_x = np.array([np.sum(df[type_var] == type_v) for type_v in type_vals])

    retrieve_t, compute_t = 0, 0

    ## See the files in a list

    ## Building the sum of local correlations
    corr_loc = np.zeros((n_vals, n_vals))
    for f in files_list:
        neighs = pd.read_csv(f, sep=';')
        indices = np.array(neighs.index)
        for j in indices:
            ## Retrieve neighs from neighs dataframe
            neighs_j = neighs.loc[j, 'neighs'].split(',')
            neighs_j = [int(e) for e in neighs_j]
            vals = df.loc[neighs_j, type_var]
            ## Count the number of companies of each type
            counts_j = np.array([np.sum(vals == val) for val in type_vals])
            cnae_val = df.loc[j, type_var]
            idx = type_vals.index(cnae_val)
            ## Compute the correlation contribution
            counts_j[idx] -= 1
            if counts_j[idx] == counts_j.sum():
                corr_loc_j = np.zeros(n_vals)
                corr_loc_j[idx] = counts_j[idx]/counts_j.sum()
            else:
                corr_loc_j = counts_j/(counts_j.sum()-counts_j[idx])
                corr_loc_j[idx] = counts_j[idx]/counts_j.sum()
            ## Aggregate to local correlation
            corr_loc[idx, :] += corr_loc

    ## Building the normalizing constants
    C = np.zeros((n_vals, n_vals))
    for i in range(n_vals):
        for j in range(n_vals):
            if i == j:
                C[i, j] = (N_t-1)/(Nx[i]*(Nx[i]-1))
            else:
                C[i, j] = (N_t-Nx[i])/(Nx[i]*Nx[j])

    ## Building a net
    net = np.log10(np.multiply(C, corr_loc))
    return net, type_vals, N_x
