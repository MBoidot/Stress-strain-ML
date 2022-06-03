import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import statsmodels.api as sm
from matplotlib.widgets import RangeSlider
import numpy as np
import piecewise_regression as pr
from scipy.interpolate import CubicSpline
from os import listdir
from os.path import isfile, join
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import keras.api._v2.keras as keras
import tensorflow as tf
from keras import datasets, layers, models, Model
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Embedding, Input, Flatten, Concatenate, Dense

# imports from current project
from Densification_threshold import getIndexes

df_input = pd.read_excel(r"./../INPUT.xlsx")
df_input = df_input.fillna(0)

oe_style = OneHotEncoder()
oe_results = oe_style.fit_transform(np.asarray(df_input["CT"]).reshape(-1, 1))
pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_)
df_input = df_input.join(
    pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_)
)

df_input = df_input.drop("CT", axis=1)

# for file in "list of files in folder"
folder = "./../0-Raw files/"
onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
df_dict = {}
print(f"nombre de fichiers a traiter : {len(onlyfiles)}")

full_ds = []
output = []

for file in onlyfiles:
    file_name = file.split(".")[0]  # only name
    df = pd.read_excel(f"{folder}{file}", sheet_name=[3, 4, 5], header=None)

    for idx, sheet in enumerate(df):
        # create a separate Df for each assay and put in a dict

        df_dict[f"meta_{sheet}"] = df[sheet].iloc[
            0 : getIndexes(df[sheet], "Début des courbes")[0][0] - 1, 1
        ]
        df_dict[f"data_{sheet}"] = df[sheet].iloc[
            getIndexes(df[sheet], "Début des courbes")[0][0] + 1 :, [2, 5, 12, 14]
        ]

        columns = df_dict[f"data_{sheet}"].iloc[0]
        df_dict[f"data_{sheet}"].columns = columns
        df_dict[f"data_{sheet}"] = df_dict[f"data_{sheet}"][2:]

    df_concat = pd.concat(
        df_dict[f"data_{sheet}"].iloc[:, [2, 3]] for sheet in df.keys()
    )
    by_row_index = df_concat.groupby(df_concat.index)
    df_stdev = by_row_index.std()
    df_means = by_row_index.mean()

    df_final = pd.concat(
        [df_means.iloc[:, 0], df_means.iloc[:, 1], df_stdev.iloc[:, 1]],
        ignore_index=True,
        axis=1,
    )

    df_final.columns = ["Strain", "Stress", "std_dev"]

    x = df_final["Strain"]
    y = df_final["Stress"]
    err = df_final["std_dev"]

    # Etape finale - compiler les données de déplacement avec les métadonnées + les données calculées
    # avec ce "vecteur" on alimentera une base qui permettra l'apprentissage
    dataset_sample = df_final.dropna()
    dataset_sample = df_final.drop("abs_err", axis=1)

    strain = dataset_sample.Strain.tolist()[0:14000]
    stress = dataset_sample.Stress.tolist()[0:14000]

    # Cubic spline works well. No need of implementation for the moment
    max_x = max(x)
    cs = CubicSpline(x, y)
    xs = np.linspace(0, max_x, 30)

    # create list of input data + stress and strain

    sample_input_list = []

    for input in range(0, df_input.shape[1]):
        sample_input_list.append(
            df_input[df_input["File"] == file_name].iloc[:, input].values
        )

    sample_input_list.pop(-4)
    sample_input_list.append(strain)
    sample_input_list.append(stress)

    sample_input_array = np.array(sample_input_list, dtype=object)
    full_ds.append(sample_input_array)

# array of lists of lists
full_array = np.array(full_ds, dtype=object)

# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]
