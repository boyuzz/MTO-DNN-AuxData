#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%%
def func(x, a, b):
    return a*np.tanh(b*x)
#%%
# xdata = np.linspace(0, 1, 200)

# y = func(xdata, 2.5, 1.3, 0.5)
# np.random.seed(1729)
# y_noise = 0.2 * np.random.normal(size=xdata.size)
# ydata = y + y_noise
# plt.plot(xdata, ydata, 'b-', label='data')
ydata = np.array([0.9228571428571428, 0.925, 0.9264285714285714, 0.9292857142857143, 0.93, 0.93, 0.9357142857142857, 0.9385714285714286])
xdata = np.arange(len(ydata))
popt, pcov = curve_fit(func, xdata, ydata)
popt

# plt.plot(np.linspace(0, 7, 200), ydata, 'r-')
plt.plot(np.linspace(0, 7, 200), func(np.linspace(0, 7, 200), *popt), 'g--',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()