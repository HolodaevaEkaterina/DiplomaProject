import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import pandas as pd

ax = plt.subplot(1, 1, 1)

# Draw the plot
#ax.bar([3, 3, 3, 4, 5,4,  4,  5, 1, 2, 1,2, 2, 4, 1, 2, 3, 4, 5],
  #      color='blue', edgecolor='black', align ='center', height = 1)

# Title and labels
table3 = pd.read_csv("Ekg_table_full.csv", ";")

table3 = table3.drop_duplicates("Code").Age


table3.plot.hist()
#plt.bar( table3["Age"], width=0.7)
#plt.xlabel('Значение', size=22)
#plt.ylabel('Количество', size=22)


#plt.subplot([3, 3, 3, 4, 5,4,  4,  5, 1, 2, 1,2, 2, 4, 1, 2, 3, 4, 5], bins=7)
plt.xlabel('Возраст', size=14)
plt.ylabel('Количество', size=14)
plt.show()