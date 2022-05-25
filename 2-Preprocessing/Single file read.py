import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.widgets import RangeSlider

import numpy as np
import matplotlib.pyplot as plt
import piecewise_regression as pr


path = ""
df = pd.read_excel(
    r"./../0-Raw files//kelvin v5 raw datas.xlsx", sheet_name=[3, 4, 5], header=None
)


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


cm = 1 / 2.54
fig, ax = plt.subplots(2, 3, sharey="row", figsize=(15 * cm, 5 * cm))

# adjust subplots to make room for the slider
plt.subplots_adjust(
    left=0.098, bottom=0.20, right=0.97, top=0.95, wspace=0.07, hspace=0.15
)


fig.supylabel("Stress")

# Define initial parameters for regression
low_strain = 2
high_strain = 4
df_dict = {}
for idx, sheet in enumerate(df):
    # create a separate Df for each assay and put in a dict

    df_dict[f"meta_{sheet}"] = df[sheet].iloc[
        0 : getIndexes(df[sheet], "Début des courbes")[0][0] - 1, 1
    ]
    df_dict[f"data_{sheet}"] = df[sheet].iloc[
        getIndexes(df[sheet], "Début des courbes")[0][0] + 1 :, [2, 5, 12, 14]
    ]
    df_dict[f"data_{sheet}"].columns = df_dict[f"data_{sheet}"].iloc[0]
    df_dict[f"data_{sheet}"] = df_dict[f"data_{sheet}"][2:]

    # obtain m (slope) and b(intercept) of linear regression line
    x = pd.to_numeric(df_dict[f"data_{sheet}"].iloc[:, 2])
    y = pd.to_numeric(df_dict[f"data_{sheet}"].iloc[:, 3])

    x_reduced = x[(x > 2) & (x < 4)]
    x_reduced = sm.add_constant(x_reduced)
    y_reduced = y[(y.index >= min(x_reduced.index)) & (y.index <= max(x_reduced.index))]

    # ordinary least squares
    model = sm.OLS(y_reduced, x_reduced, missing="drop")
    results = model.fit()

    ax[1, idx].plot(
        x_reduced.iloc[:, 1],
        results.params[0] * x_reduced.iloc[:, 1] + results.params[1],
    )
    ax[1, idx].plot(x_reduced.iloc[:, 1], y_reduced)
    ax[1, idx].text(
        0, 0, f"{results.params[0]:.3e}, {results.params[1]:.3e}", fontsize=9
    )
    ax[1, idx].set_xlabel("Strain (%)")

plt.legend(fontsize=9)
plt.show()
