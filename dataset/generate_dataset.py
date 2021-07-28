import sklearn.datasets as dt
import matplotlib.pyplot as plt
import numpy as np

seed = 1
# Create dataset
"""
x_data,y_data = dt.make_classification(n_samples=1000,
                                       n_features=2,
                                       n_repeated=0,
                                       class_sep=2,
                                       n_redundant=0,
                                       random_state=seed)
"""

x_data,y_data = dt.make_circles(n_samples=500,
                              noise=0.2,
                              factor=0.3)

# Plot dataset
my_scatter_plot = plt.scatter(x_data[:,0],
                                  x_data[:,1],
                                  c=y_data,
                                  vmin=min(y_data),
                                  vmax=max(y_data),
                                  s=35)
plt.savefig("data.png")
plt.show()



# Format y_data
y_data = np.array([[1,0] if y==0 else [0,1] for y in y_data])

# Save data into csv files
np.savetxt("x_data.csv", x_data,delimiter=',',fmt='%f')
np.savetxt("y_data.csv", y_data,delimiter=',',fmt='%f')


