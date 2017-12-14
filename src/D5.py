#%%
import numpy as np 
import pandas as pd
from pandas import Series, DataFrame

#%%
a = np.random.rand(5)
print('a is an array:', a)
s = Series(a)
print('s is a Series:\n', s)

#%%
s = Series(np.random.randn(5), 
           index=['a','b','c','d','e'],
           name='my_series')
print(s)
print(s.index)
print(s.name)

#%%
d = {'a': 0., 'b': 1, 'c': 2}
print('d is a dict:\n', d)
s = Series(d)
print('s is a Series:\n', s)

#%%
s = Series(np.random.randn(10), index=['a','b','c','d','e','f','g','h','i','j'])
s[0]
s[:2]
s[[2,0,4]]
s[['e', 'i']]
s[s > 0.5]

#%%
d = {'one': Series([1.,2.,3.], index=['a','b','c']),
     'two': Series([1.,2.,3.,4.], index=['a','b','c','d'])
}
df = DataFrame(d)
print(df)

df = DataFrame(d, index=['r','d','a'], 
               columns=['two', 'three'])
print(df)
print('DataFrame index:\n', df.index)
print('DataFrame columns:\n', df.columns)
print('DataFrame values:\n', df.values)

#%%
d = {'one': [1.,2.,3.,4.], 'two': [4.,3.,2.,1.]}
df = DataFrame(d, index=['a','b','c','d'])
df

#%%
d = [{'a': 1.6, 'b': 2}, {'a': 3, 'b': 6, 'c': 9}]
df = DataFrame(d)
df

#%%
a = Series(range(5))
b = Series(np.linspace(4, 20, 5))
df = pd.concat([a, b], axis=1)
df

#%%
df = DataFrame()
index = ['alpha', 'beta', 'gamma', 'delta', 'eta']
for i in range(5):
    a = DataFrame([np.linspace(i, 5*i, 5)], index=[index[i]])
    df = pd.concat([df, a], axis=0)
df

#%%
# print(df[1])
# print(type(df[1]))
df.columns = ['a','b','c','d','e']
print(df['b'])
print(type(df['b']))
print(df.b)
print(type(df.b))
print(df[['a','d']])
print(type(df[['a','d']]))
print(df['b'][2])
print(df['b']['gamma'])
print(df.iloc[1])
print(df.loc['beta'])

#%%
print('Selecting by slices:')
print(df[1:3])

bool_vec = [True, False, True, True, False]
print('Selectiing by boolean vector:')
print(df[bool_vec])

#%%
print(df[['b', 'd']].iloc[[1, 3]])
print(df.iloc[[1, 3]][['b', 'd']])
print(df[['b', 'd']].loc[['beta', 'delta']])
print(df.loc[['beta', 'delta']][['b', 'd']])
print(df.iat[2, 3])
print(df.at['gamma', 'd'])
