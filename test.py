import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 输入您的数组数据
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
import pandas as pd
pd_data = pd.DataFrame(data=data, columns=["data"])
sns.histplot(data,bins=30,kde=False)
plt.show()