# python -m pip install scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

draws_10 = stats.norm.rvs(size=10000, loc=0, scale=10 )
draws_20 = stats.norm.rvs(size=10000, loc=0, scale=20 )
draws_30 = stats.norm.rvs(size=10000, loc=0, scale=30 )


plt.subplot(3,1,1)
plt.hist(draws_10, 50)
plt.axis([-100, 100, 0, 800])

plt.subplot(3,1,2)
plt.hist(draws_20, 50)
plt.axis([-100, 100, 0, 800])


plt.subplot(3,1,3)
plt.hist(draws_30, 50)
plt.axis([-100, 100, 0, 800])

plt.show()