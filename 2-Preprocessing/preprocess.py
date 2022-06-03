import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.interpolate import CubicSpline
from os import listdir
from os.path import isfile, join
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import keras.api._v2.keras as keras
from sklearn.preprocessing import OneHotEncoder
from itertools import chain
import random
import sklearn.neural_network as nn
import sklearn.preprocessing as pp


def getIndexes(dfObj, value):
    """Get index positions of value in dataframe i.e. dfObj."""
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos


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
# set the number of interpolation points
inter_points = 30
num_parameters = df_input.shape[1] - 1

df_dict = {}
f = 0
# print(f"nombre de fichiers a traiter : {len(onlyfiles)}")
full_ds = []


for file in onlyfiles:
    file_name = file.split(".")[0]  # only name
    # print(f"Treatment number {f+1}/{len(onlyfiles)}")

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

    # Etape finale - compiler les données de déplacement avec les métadonnées + les données calculées
    # avec ce "vecteur" on alimentera une base qui permettra l'apprentissage
    dataset_sample = df_final.dropna()
    x = dataset_sample["Strain"]
    x_sm = x[::20]
    y = dataset_sample["Stress"]
    y_sm = y[::20]
    err = dataset_sample["std_dev"]
    err_sm = err[::20]

    # Cubic spline works well. No need of implementation for the moment

    max_x = max(x_sm)
    cs = CubicSpline(x_sm, y_sm)
    xs = np.linspace(0, max_x, inter_points)

    # create list of input data + stress and strain

    sample_input_list = []

    for input in range(0, df_input.shape[1]):
        sample_input_list.append(
            df_input[df_input["File"] == file_name].iloc[:, input].values.tolist()
        )

    sample_input_list.pop(-(len(oe_style.categories_[0]) + 1))
    # print(len(sample_input_list))
    # print(sample_input_list)
    sample_input_list.append(xs.tolist())
    # print(sample_input_list)

    cubic_y = cs(xs).tolist()
    # print(len(cubic_y))
    sample_input_list.append(cubic_y)
    # print(sample_input_list)
    # print(len(sample_input_list))
    sample_input_list = list(map(float, chain.from_iterable(sample_input_list)))
    # print(sample_input_list)
    print(f"file being processed : {file_name}, vector length {len(sample_input_list)}")
    full_ds.append(sample_input_list)

    plt.plot(xs, cs(xs))
    plt.ylim(0, max(cs(xs)))
    plt.fill_between(x_sm, y_sm - err_sm, y_sm + err_sm, alpha=0.2)
    plt.title(label=file_name)
    plt.xlabel("Strain")
    plt.ylabel("Stress")
    plt.savefig(f"output/{file_name}.jpg", dpi=300)
    plt.show()

    f += 1

# array of lists of lists
full_ds = list(map(float, chain.from_iterable(full_ds)))

full_array = np.array(full_ds, dtype=object)
full_array = full_array.reshape(len(onlyfiles), num_parameters + 2 * inter_points)

x = full_array[:, 0:num_parameters]
y = full_array[:, num_parameters + inter_points + 1 : len(full_array[0])]
y = y.astype("float32")

random.seed(10)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

scaler = pp.RobustScaler()
scaler.fit(X_train)
xtrain = scaler.transform(X_train)
xtest = scaler.transform(X_test)
model = nn.MLPRegressor(
    hidden_layer_sizes=(
        num_parameters + inter_points,
        2 * (num_parameters + inter_points),
        num_parameters + inter_points,
    )
)
model.fit(X_train, y_train)
score = model.score(X_train, y_train)
print(score)
ypred = model.predict(X_test)


# compare prediction and train data

for i in np.arange(len(ypred)):

    plt.plot(np.linspace(0, 100, inter_points - 1), ypred[i])
    plt.show()


# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]
