import numpy as np
from scipy.stats import ttest_ind

v1 = np.random.normal(size=100)
v2 = np.random.normal(size=100)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
res = ttest_ind(v1, v2)

print(res)
