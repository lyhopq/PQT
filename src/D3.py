#%%
import numpy as np 

#%%
a = np.arange(20)
print(a)
a = a.reshape(4, 5)
print(a)
a = a.reshape(2, 2, 5)
print(a)

#%%
a.ndim
a.shape
a.size
a.dtype

#%%
raw = [0,1,2,3,4]
a = np.array(raw)
a

#%%
raw = [[0,1,2,3,4], [5,6,7,8,9]]
b = np.array(raw)
b

#%%
np.zeros((4,5))
np.ones((4,5), dtype=int)

#%%
np.random.rand(5)

#%%
a = np.array([[1.0, 2], [2, 4]])
print('a:\n', a)
b = np.array([[3.2, 1.5], [2.5, 4]])
print('b:\n', b)
print('a+b:\n', a+b)
print('3*a:\n', 3*a)
print('b+1.8:\n', b+1.8)
a /= 2
print('a /= 2:\n', a)
print('np.exp(a):\n',  np.exp(a))
print('np.sqrt(a):\n', np.sqrt(a))
print('np.square(a):\n', np.square(a))
print('np.power(a,3):\n', np.power(a,3))

#%%
a = np.arange(20).reshape(4,5)
print('a:\n', a)
print('sum:', a.sum())
print('maximum element in a:', a.max())
print('minimun element in a:', a.min())
print('maximum elements in each row of a:', a.max(axis=1))
print('minimum elements in each column of a:', a.max(axis=1))

#%%
a = np.arange(20).reshape(4,5)
a = np.asmatrix(a)
print(type(a))

b = np.matrix('1.0 2.0; 3.0 4.0')
print(type(b))

#%%
a = np.arange(20).reshape(4,5)
a = np.mat(a)
print('a:\n', a)
b = np.arange(2, 45, 3).reshape(5, 3)
b = np.mat(b)
print('b:\n', b)
print('a*b:\n', a*a)

#%%
a = np.array([[3.2,1.5], [2.5,4]])
print(a[0][1], a[0,1])

b = a
a[0][1] = 2.0
print('a:\n', a)
print('b:\n', b)

#%%
a = np.array([[3.2,1.5], [2.5,4]])
b = a.copy()
a[0][1] = 2.0
print('a:\n', a)
print('b:\n', b)

#%%
a = np.array([[3.2,1.5], [2.5,4]])
b = a
a = np.array([[2,1], [9,3]])
print('a:\n', a)
print('b:\n', b)

#%%
a = np.arange(20).reshape(4,5)
print('a:\n', a)
print('the 2nd and 4th column of a:\n', a[:, [1,3]])
print('a[:,2][a[:, 0] > 5:\n', a[:, 2][a[:, 0] > 5])
loc = np.where(a==11)
print(loc)
print(a[loc[0][0], loc[0][0]])

#%%
#p36