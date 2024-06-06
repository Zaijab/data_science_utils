"""
This file contains a number of functions which operate 
"""


def list_loc(df, index_list, complement=False):
    return (
        df.loc[df.index.isin(index_list)]
        if not complement
        else df.loc[~df.index.isin(index_list)]
    )
