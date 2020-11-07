import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import os

data = pd.read_csv("cancer.csv")
#data.drop(['id'], inplace = True, axis = 1)

data.Bare_Nuclei = pd.to_numeric(data.Bare_Nuclei, errors='coerce')

data.hist(column='id')
data.hist(column='Clump_Thickness')
data.hist(column='Uniformity_Cell_Size')
data.hist(column='Uniformity_Cell_Shape')
data.hist(column='Marginal_Adhesion')
data.hist(column='Single_Epithelial_Size')
data.hist(column='Bare_Nuclei')
data.hist(column='Bland_Chromatin')
data.hist(column='Normal_Nucleoli')
data.hist(column='Mitoses')
data.hist(column='Class')
