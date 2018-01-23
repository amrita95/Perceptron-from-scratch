import numpy as np
import LoadingData as data

dat = data.trdata

z = np.arange(0,10)


a = []
b=[]
b.append([1,2])
b.append([5,4])
a.append(2)
a.append(4)

print(b)
print(((a[1])*np.mat(b)[1]))



