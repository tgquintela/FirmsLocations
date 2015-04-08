
"""
"""

def computation_neighbourhood(i, locations, types):
    """"""
    pass


def self_interaction(type_, df, loc_vars, type_var):
    """"""
    ## 0. Computation of needed variables
    idxs = get_from_type(df, type_var)
    N_t = df.shape[0]
    N_a = idxs.shape[0]

    ## 1. Computation of the index
    C = np.log10((N_t-1)/float(N_a*(N_a-1)))
    suma = 0
    for i in idxs:
        n_a, n_t = computation_neighbourhood(i, df, types_)
        suma = suma + n_a/float(n_t)
    a_AA = C*suma
    return a_AA


def x_interaction(type_, df, loc_vars, type_vars):
    """"""
    ## 0. Computation of needed variables
    idxs_A = get_from_type(df, type_var[0])
    idxs_B = get_from_type(df, type_var[1])

    N_t = locations.shape[0]
    N_a = idxs_A.shape[0]
    N_b = idxs_B.shape[0]

    ## 1. Computation of the index
    C = np.log10((N_t-N_a)/float(N_a*N_b))
    suma = 0
    for i in idxs:
        ns, n_t = computation_neighbourhood(i, df, types_)
        n_a, n_b = ns
        suma = suma + n_b/float((n_t-n_a))

    a_AB = C*suma
    return a_AB
