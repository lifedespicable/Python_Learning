from pandas import Series, DataFrame
import pandas as pd

# obj = Series([4, 5, 6 , -7])
# print(obj)
# print(obj.shape)
# print(type(obj))
# print(obj.dtype)
# print(obj.index)
# print(obj.values)
#
# obj2 = Series([4 ,7 , -5, 3], index=['d', 'b','c', 'a'])
# print(obj2)
# print(obj.shape)
# print(type(obj2))
# print(obj2.dtype)
# print(obj2.index)
# print(obj2.values)
#
# obj2['c'] = 6
# print(obj2)
#
# print('a' in obj2)
#
# sdata = {'beijing': 35000, 'shanghai': 71000, 'guangzhou': 16000, 'shenzhen': 5000}
# obj3 = Series(sdata)
# print(obj3)
#
# obj3.index = ['bj', 'sh' ,'gz' , 'sz']
# print(obj3)

data = {'city': ['shanghai', 'shanghai', 'shanghai', 'beijing', 'beijing'],
        'year': [2016, 2017, 2018, 2017, 2018],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
print(frame)
# print(frame.shape)
# print(type(frame))
# print(frame.dtypes)
# print(frame.values)

frame2 = DataFrame(data, columns= ['year', 'city', 'pop'])
print(frame2)
print(frame2['year'])
print(frame2.year)
print(frame2['city'])
print(frame2.city)

frame2['new'] = 100
print(frame2)
frame2['cap'] = (frame2.city == 'beijing')
print(frame2)

pop ={'beijing': {2008: 1.5, 2009: 2.0},
      'shanghai':{2008: 2.0, 2009: 3.6}}
frame3 = DataFrame(pop)
print(frame3)
print(frame3.T)
print(frame3.shape)
print(type(frame3))
print(frame3.values)