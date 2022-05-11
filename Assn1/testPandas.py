import pandas as pd
import numpy as np

# list1 = [1, 2, 3, 4, 5]
# s = pd.Series(list1)
# print(s)
#
# s = pd.Series(list1, index=['a', 'b', 'c', 'd', 'e'])
# print(s, s['a'])


# data = range(5)
# s = pd.Series(data, index=['a', 'b', 'c', 'd', 'e'])
# print(s)
#
# data = {chr(ord('a') + i): i for i in range(5)}
# s = pd.Series(data)
# print(s)

# basic function
s = pd.Series(np.random.randint(10, size=(10)))
print(s)
print(type(s.values))
print(s.head(1))
print(s.head(-1))
print(s.tail(2))  # 倒数
