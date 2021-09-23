import numpy as np
import pandas as pd

dataset = np.load('polution_dataSet.npy')

df = pd.DataFrame(data=dataset,  columns=["Pollution", "Dew", "Temp", "Pressure", "Wind_Dir", "Wind_SPD", "Snow", "Rain"])
cov_matrix = df.cov()