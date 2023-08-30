import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cd ... # enter the current directory
df = pd.read_csv("raman_5.csv", index_col=None)
#df

# Write code to converting the values to actual ones. Comment this step if the data need not be modified.
df["Wavenumber"] = df["Wavenumber"] / 10000
df["Intensity"] = df["Intensity"] / 10000
df

def find_first_two_max_indices(X):
    local_maxima_indices = []

    for i in range(1, len(X) - 1):
        if X[i] > X[i - 1] and X[i] > X[i + 1]:
            local_maxima_indices.append(i)

    if len(local_maxima_indices) < 2:
        return None, None  # Handle cases with less than two local maxima

    sorted_indices = sorted(local_maxima_indices, key=lambda index: X[index], reverse=True)

    return [sorted_indices[0]], [sorted_indices[1]]

from scipy.optimize import curve_fit

def lorentzian(x, amplitude, center, width):
    return amplitude / (1 + ((x - center) / (0.5 * width)) ** 2)

def fit_single_lorentzian(xdata, ydata):
    initial_guess = [max(ydata), xdata[np.argmax(ydata)], 1.0]  # Initial guesses for amplitude, center, and width

    popt, _ = curve_fit(lorentzian, xdata, ydata, p0=initial_guess)

    fit_ydata = lorentzian(xdata, *popt)

    return popt, fit_ydata

# code for fitting both peaks, and returning the wavenumbers, intensities, fitting parameters
def peak_fitting(xdata, ydata):
  peak_indices = find_first_two_max_indices( ydata ) # 1 is added to index because the STOP number is not counted.
  first_indices, second_indices = peak_indices[0], peak_indices[1]
  wavenum2_list = [] # list containing wavenumbers in the proximity of the second peak
  inten2_list = [] # list containing intensity in the proximity of the second peak
  for wavenumber in xdata:
    if wavenumber >= 375 and wavenumber <= 395: # adjust these limits as per the limits of the second peak in the user's data
      wavenum2_list.append(wavenumber)

      xdata = list(xdata)
      index = xdata.index(wavenumber)

      inten2_list.append(ydata[index])
  popt2, fit_ydata2 = fit_single_lorentzian(wavenum2_list, inten2_list) # We need the data to have atleast 3-5
                                                        # datapoints per free parameter. So at least 9-15 points needed here.

  xdata = np.array(xdata)
  wavenum1_list = [] # list containing wavenumbers in the proximity of the first peak
  inten1_list = [] # list containing intensity in the proximity of the first peak
  for wavenumber in xdata:
    if wavenumber >= 400 and wavenumber <= 420: # adjust these limits as per the limits of the first peak in the user's data
      wavenum1_list.append(wavenumber)

      xdata = list(xdata)
      index = xdata.index(wavenumber)

      inten1_list.append(ydata[index])
  popt1, fit_ydata1 = fit_single_lorentzian(wavenum1_list, inten1_list) # We need the data to have atleast 3-5
                                                        # datapoints per free parameter. So at least 9-15 points needed here.
  return wavenum1_list, fit_ydata1, popt1, wavenum2_list, fit_ydata2, popt2

num_rows = len(df["X"])
Wavenumbers1 = [] # wavenumbers corresponding to first maximum intensities
Wavenumbers2 = [] # wavenumbers corresponding to second maximum intensities
Intensities1 = [] # values of the first maximum intensities
Intensities2 = [] # values of the second maximum intensities
X_list = [] # x coordinates corresponding to first and second maximum intensities
Y_list = [] # y coordinates corresponding to first and second maximum intensities

coordinate_bool = True # Unnecessary. It is True when the coordinate corresponding to the current index is same as that of the next index. It is False when not
change_indices = [] # indices for which the corresponding coordinate is not same as the coordinate for the index 1 greater than this one.
for index in range(num_rows-1): # We subtract 1 because we are using 'index + 1' as an index in the code below
  if (df["X"])[index] == (df["X"])[index+1] and (df["Y"])[index] == (df["Y"])[index+1]: # i.e. if we are at the same (x, y) point
                                                                                        # as compared with the prev iteration
    coordinate_bool = True

  else:
    coordinate_bool = False
    change_indices.append(index)

for index in range(len(change_indices)): # indexing the list change_indices
  if index == 0:
    xdata = np.array(list(df["Wavenumber"])[:change_indices[index]+1])
    ydata = np.array(list(df["Intensity"])[:change_indices[index]+1])
    ydata = ydata - np.min(ydata)

    wavenum1_list = peak_fitting(xdata, ydata)[0]
    fit_ydata1 = peak_fitting(xdata, ydata)[1]
    popt1 = peak_fitting(xdata, ydata)[2]
    wavenum2_list = peak_fitting(xdata, ydata)[3]
    fit_ydata2 = peak_fitting(xdata, ydata)[4]
    popt2 = peak_fitting(xdata, ydata)[5]

    xdata = list(xdata) # redundant step
    ydata = list(ydata) # redundant step

    Wavenumbers1.append(popt1[1])
    Intensities1.append(popt1[0])
    Wavenumbers2.append(popt2[1])
    Intensities2.append(popt2[0])
    X_list.append(list(df["X"])[change_indices[index]]) # all position coordinates are same in this iteration. So I choose one
    Y_list.append(list(df["Y"])[change_indices[index]]) # all position coordinates are same in this iteration. So I choose one

  elif index == len(change_indices) - 1: # i.e. the maximum value of index. we use 'elif' instead of 'if' to avoid errors for small datasets
                                        # when both first and second will hold simultaneously.
    xdata = np.array(list(df["Wavenumber"])[change_indices[index]+1 : ]) # 1 is added to index because the STOP number is not counted.
    ydata = np.array(list(df["Intensity"])[change_indices[index]+1 : ])
    ydata = ydata - np.min(ydata)

    wavenum1_list = peak_fitting(xdata, ydata)[0]
    fit_ydata1 = peak_fitting(xdata, ydata)[1]
    popt1 = peak_fitting(xdata, ydata)[2]
    wavenum2_list = peak_fitting(xdata, ydata)[3]
    fit_ydata2 = peak_fitting(xdata, ydata)[4]
    popt2 = peak_fitting(xdata, ydata)[5]

    xdata = list(xdata) # redundant step
    ydata = list(ydata) # redundant step

    Wavenumbers1.append(popt1[1])
    Intensities1.append(popt1[0])
    Wavenumbers2.append(popt2[1])
    Intensities2.append(popt2[0])
    X_list.append(list(df["X"])[change_indices[index]]) # all position coordinates are same in this iteration. So I choose one
    Y_list.append(list(df["Y"])[change_indices[index]]) # all position coordinates are same in this iteration. So I choose one

  else: # when index is not the first or last index
    xdata = np.array(list(df["Wavenumber"])[change_indices[index-1]+1 : change_indices[index]+1 ])
    ydata = np.array(list(df["Intensity"])[change_indices[index-1]+1 : change_indices[index]+1 ])
    ydata = ydata - np.min(ydata)

    wavenum1_list = peak_fitting(xdata, ydata)[0]
    fit_ydata1 = peak_fitting(xdata, ydata)[1]
    popt1 = peak_fitting(xdata, ydata)[2]
    wavenum2_list = peak_fitting(xdata, ydata)[3]
    fit_ydata2 = peak_fitting(xdata, ydata)[4]
    popt2 = peak_fitting(xdata, ydata)[5]

    xdata = list(xdata) # redundant step
    ydata = list(ydata) # redundant step

    Wavenumbers1.append(popt1[1])
    Intensities1.append(popt1[0])
    Wavenumbers2.append(popt2[1])
    Intensities2.append(popt2[0])
    X_list.append(list(df["X"])[change_indices[index]]) # all position coordinates are same in this iteration. So I choose one
    Y_list.append(list(df["Y"])[change_indices[index]]) # all position coordinates are same in this iteration. So I choose one

import pandas as pd

# Example lists
list1 = X_list
list2 = Y_list
list3 = Wavenumbers1
list4 = Intensities1
list5 = Wavenumbers2
list6 = Intensities2

# Creating a DataFrame
data2 = {'X': list1, 'Y': list2, 'Wavenumber1': list3, 'Intensity1': list4, 'Wavenumber2': list5, 'Intensity2': list6}
df2 = pd.DataFrame(data2)

# Wavenumber separation
x = []
y = []
z = []
for i in range(len(X_list)):
  x.append(X_list[i])
  y.append(Y_list[i])
  z.append(Wavenumbers1[i]-Wavenumbers2[i]) # Finding the wavenumber separation

from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize

x_ = x  # Array 1
y_ = y   # Array 2
z_ = z      # Array 3

triang = Triangulation(x_, y_)
fig, ax = plt.subplots()

# Create a continuous color plot using tricontourf
contour = ax.tricontourf(triang, z_, levels=45, cmap='autumn')  # You can use any colormap you prefer
cbar = plt.colorbar(contour)
cbar.set_label('separation between peaks (in 1/cm)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Continuous Color Plot')
plt.show()

# saving all the possible colormap formats in a single directory
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri

save_directory = ... # Specify the directory to save figures

x_ = x
y_ = y
z_ = z

triang = mtri.Triangulation(x_, y_)
colormaps = plt.colormaps()
rows = len(colormaps) // 3 + 1
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))

for i, cmap in enumerate(colormaps):
    ax = axes[i // cols, i % cols]
    contour = ax.tricontourf(triang, z_, levels=15, cmap=cmap)
    ax.set_title(cmap)
    ax.axis('off')
    fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.05, shrink=0.99)

    filename = os.path.join(save_directory, f'colormap_{cmap}.png')
    plt.savefig(filename)
    plt.clf()

plt.tight_layout()
plt.show()
