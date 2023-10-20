from sklearn.model_selection import train_test_split
from ELLA import ELLA

from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from scipy.linalg import norm
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, explained_variance_score
from matplotlib.pyplot import MultipleLocator

def multi_task_train_test_split(Xs,Ys,train_size=0.5):
    Xs_train = []
    Ys_train = []
    Xs_test = []
    Ys_test = []
    for t in range(len(Xs)):
        X_train, X_test, y_train, y_test = train_test_split(Xs[t], np.squeeze(Ys[t]), train_size=train_size)
        Xs_train.append(X_train)
        Xs_test.append(X_test)
        Ys_train.append(y_train)
        Ys_test.append(y_test)
    return Xs_train, Xs_test, Ys_train, Ys_test





T = 20
d = 10
n = 100
k = 5
noise_var = .1


model = ELLA(d,k,Ridge,mu=1,lam=10**-5)

S_true = np.random.randn(k,T)
L_true = np.random.randn(d,k)
w_true = L_true.dot(S_true)

# make sure to add a bias term (it is not done automatically)
Xs = [np.hstack((np.random.randn(n,d-1), np.ones((n,1)))) for i in range(T)]
# generate the synthetic labels
Ys = [Xs[i].dot(w_true[:,i]) + noise_var*np.random.randn(n,) for i in range(T)]
# break into train and test sets
Xs_train, Xs_test, Ys_train, Ys_test = multi_task_train_test_split(Xs,Ys,train_size=0.5)


a = []
b = []
for t in range(T):
    model.fit(Xs_train[t], Ys_train[t], t)
    single_task_model = Ridge(fit_intercept=False, ).fit(Xs_train[t], Ys_train[t])
    a.append(explained_variance_score(single_task_model.predict(Xs_test[t]), Ys_test[t]))
    b.append(model.score(Xs_test[t], Ys_test[t], t))

print(a)
print(b)
plt.figure()
plt.plot(list(range(T)), a, 'r-', linewidth=1, alpha=1)
plt.plot(list(range(T)), b, 'b-', linewidth=1, alpha=1)
plt.legend(['Single Task score', 'ELLA score'])
plt.xlabel('Task_id')
plt.ylabel('Model_score')

X_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(X_major_locator)
plt.show()


print("Average explained variance score", np.mean([model.score(Xs_test[t], Ys_test[t][:,np.newaxis], t) for t in range(T)], axis=0))

# Try out a classification problem
Ys_binarized_train = [Ys_train[i] > 0 for i in range(T)]
Ys_binarized_test = [Ys_test[i] > 0 for i in range(T)]

model = ELLA(d,k,LogisticRegression,mu=1,lam=10**-5)
for t in range(T):
    model.fit(Xs_train[t], Ys_binarized_train[t], t)



print("Average classification accuracy", np.mean([model.score(Xs_test[t], Ys_binarized_test[t], t) for t in range(T)]))




data = loadmat('landminedata.mat')

Xs_lm = []
Ys_lm = []
for t in range(data['feature'].shape[1]):
    X_t = data['feature'][0,t]
    Xs_lm.append(np.hstack((X_t,np.ones((X_t.shape[0],1)))))
    Ys_lm.append(data['label'][0,t] == 1.0)

d = Xs_lm[0].shape[1]
k = 1

Xs_lm_train, Xs_lm_test, Ys_lm_train, Ys_lm_test = multi_task_train_test_split(Xs_lm,Ys_lm,train_size=0.5)
model = ELLA(d,k,LogisticRegression,{'C':10**0},mu=1,lam=10**-5)
for t in range(T):
    model.fit(Xs_lm_train[t], Ys_lm_train[t], t)

print(model.S)

print("Average AUC:", np.mean([roc_auc_score(Ys_lm_test[t],
                                             model.predict_logprobs(Xs_lm_test[t], t))
                               for t in range(1)]))

