import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import statsmodels.api as sm
from matplotlib.widgets import RangeSlider
import numpy as np
import matplotlib.pyplot as plt
import piecewise_regression as pr
from os import listdir
from os.path import isfile, join
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import keras.api._v2.keras as keras
import tensorflow as tf
from keras import datasets, layers, models


df_input = pd.read_excel(r"./../INPUT.xlsx")


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


# for file in "list of files in folder"
# on fait le traitement de tous les fichiers puis on mets ça dans une liste de liste qu'on convertira en np.array à la fin

folder = "./../0-Raw files/"
onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
df_dict = {}
print(f"nombre de fichiers a traiter : {len(onlyfiles)}")
f = 0

full_ds = []
output = []

cols = ["Filename", "2 to 4 slope", "Curve area", "Densification threshold"]

for file in onlyfiles:

    file_name = file.split(".")[0]  # only name
    print(file_name)
    print(f"traitement numéro : {f+1}/{len(onlyfiles)}")
    f += 1
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
    xreduced1 = x[x > 4]
    yreduced1 = y[y.index > min(xreduced1.index) + 3]
    xreduced1 = xreduced1.iloc[::100].tolist()
    yreduced1 = yreduced1.iloc[::100].tolist()

    # perform a piecewise regression on one full curve to approximate the tangent position
    nbreakpoint = 10
    pw_fit = pr.Fit(xreduced1, yreduced1, n_breakpoints=nbreakpoint)

    # on fait un tableau avec les valeurs de
    # densification threshold pour chaque indice de breakpoint
    dens_threshold_table = []
    OA_regression_table = []

    for i in range(nbreakpoint - 1):
        breakpoint_x = pw_fit.get_results()["estimates"][f"breakpoint{i+1}"]["estimate"]
        df_final["abs_err"] = abs(df_final.iloc[:, 0] - breakpoint_x)

        # ligne qui correspond au breakpoint
        df_final[df_final.abs_err == df_final.abs_err.min()]

        stress_at_breakpoint = df_final[
            df_final.abs_err == df_final.abs_err.min()
        ].iloc[0, 1]

        # linear fit between 2 and 4 percent
        x_reduced = x[(x > 2) & (x < 4)]
        y_reduced = y[
            (y.index >= min(x_reduced.index)) & (y.index <= max(x_reduced.index))
        ]

        # régression
        a, b, r, p_value, std_err = linregress(x_reduced, y_reduced)

        # ordonnée à l'origine de la droite tangente en (x_breakpoint, f(x_breakpoint))
        b2 = stress_at_breakpoint - a * breakpoint_x

        # seuil de densification
        dens_threshold_table.append(-b2 / a)
        OA_regression_table.append(b2)

    # index of suitable breakpoint
    suit_bp_idx = dens_threshold_table.index(max(dens_threshold_table)) + 1
    breakpoint_x = pw_fit.get_results()["estimates"][f"breakpoint{suit_bp_idx}"][
        "estimate"
    ]

    pw_fit.plot_fit(color="red", linewidth=1)

    dens_threshold = max(dens_threshold_table)
    b2 = min(OA_regression_table)
    print(f"dens_threshold = {dens_threshold}")
    # integration
    # on full dataset
    idx = np.where((np.array(x) >= 0) & (np.array(x) <= breakpoint_x))[0]
    integral_full = np.trapz(x=np.array(x)[idx], y=np.array(y)[idx])

    # on reduced dataset
    xreduced2 = x.iloc[::100].tolist()
    yreduced2 = y.iloc[::100].tolist()
    idxr = np.where((np.array(xreduced2) >= 0) & (np.array(xreduced2) <= breakpoint_x))[
        0
    ]
    integral_reduced = np.trapz(
        x=np.array(xreduced2)[idxr], y=np.array(yreduced2)[idxr]
    )
    output.append([file_name, a, integral_full, dens_threshold])

    plt.plot(x, y)
    plt.plot(x, a * x + b)
    plt.plot(x, a * x + b2)
    plt.ylim(0, max(y))
    plt.fill_between(x, y - err, y + err, alpha=0.2)
    plt.fill_between(x, y, where=(0 < x) & (x < dens_threshold), color="b", alpha=0.2)

    plt.title(label=file_name)
    plt.xlabel("Strain")
    plt.ylabel("Stress")
    # plt.xaxis.set_major_locator(MultipleLocator(20))
    # plt.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # plt.xaxis.set_minor_locator(MultipleLocator(5))

    # plt.yaxis.set_major_locator(MultipleLocator(0.5))
    # plt.yaxis.set_minor_locator(MultipleLocator(0.1))

    # plt.xaxis.grid(True,'minor')
    # plt.yaxis.grid(True,'minor')
    # plt.xaxis.grid(True,'major',linewidth=2)
    # plt.yaxis.grid(True,'major',linewidth=2)

    plt.savefig(f"output/{file_name}.jpg", dpi=300)
    plt.show()

    # faire un autre diagramme avec le zoom sur les données de 2 à 4%
    plt.plot(x_reduced, y_reduced)
    plt.plot(x_reduced, a * x_reduced + b)
    plt.title(label=file_name)
    plt.xlabel("Strain")
    plt.ylabel("Stress")
    plt.savefig(f"output/{file_name}_reduced.jpg", dpi=300)
    plt.show()

    # Etape finale - compiler les données de déplacement avec les métadonnées + les données calculées
    # avec ce "vecteur" on alimentera une base qui permettra l'apprentissage
    dataset_sample = df_final
    dataset_sample = df_final.drop("abs_err", axis=1)

    strain = dataset_sample.Strain.tolist()
    stress = dataset_sample.Stress.tolist()

    # create list of input data + stress and strain

    sample_input_list = []
    for input in range(0, df_input.shape[1]):
        sample_input_list.append(
            df_input[df_input["File"] == "Fluorite v1"].iloc[:, input].values.tolist()
        )

    sample_input_list.append(strain)
    sample_input_list.append(stress)

    full_ds.append(sample_input_list)

full_array = np.array(full_ds)

df1 = pd.DataFrame(output, columns=cols)
df1.to_csv("output/out.csv", index=False)

#training model
model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_dim=9),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

y = full_array[:,-1]
x = full_array[:,np.arange(0,full_array.shape[1]-1)]

model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
model.summary()

history = model.fit(x, y, epochs=10, batch_size=1, validation_split=0.2)
eval = model.evaluate(x, y)
print(eval)