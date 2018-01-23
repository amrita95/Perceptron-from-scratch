import LoadingData as data
from sklearn.metrics import zero_one_loss
from Functions2 import perceptron,predict,norm
import numpy as np
import matplotlib.pyplot as plt

iter = np.arange(0,1000,50)

trerror = np.zeros((len(iter),1))
tterror = np.zeros((len(iter),1))
predtrainlabels = np.zeros((len(iter),len(data.trlabels)))
predtestlabels = np.zeros((len(iter),len(data.ttlabels)))

for i in range(0,len(iter)):
    print(i)
    weight = perceptron(norm(data.trdata,2), data.trlabels, maxiter=iter[i])

    predtestlabels[i,:] = predict(norm(data.ttdata,2), weight).transpose()
    predtrainlabels[i,:] = predict(norm(data.trdata,2), weight).transpose()
    trerror[i]=(zero_one_loss(data.trlabels, predtrainlabels[i].transpose()))
    tterror[i]=(zero_one_loss(data.ttlabels, predtestlabels[i].transpose()))


plt.figure(1)
plt.title('Perceptron for Logistic Regression')
plt.xlabel('Iterations')
plt.ylabel('Error Rate')
plt.plot(iter, trerror, 'b', label='Training error')
plt.plot(iter, tterror, 'g', label='Testing error')
plt.legend()

plt.show()

