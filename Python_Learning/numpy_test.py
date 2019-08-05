import numpy as np

# arr1 = np.array([2,3,4])
# print(arr1)
# print(arr1.shape)
# print(type(arr1))
# print(arr1.dtype)
#
# arr2 = np.array([1.2, 2.3, 3.4])
# print(arr2)
# print(arr2.shape)
# print(type(arr2))
# print(arr2.dtype)
#
# arr3 = arr1 + arr2
# print(arr3)
# print(arr3.shape)
# print(type(arr3))
# print(arr3.dtype)
#
# arr4 = arr2 * 10
# print(arr4)
# print(arr4.shape)
# print(type(arr3))
# print(arr4.dtype)
#
# data = [[1, 2, 3],[4, 5, 6]]
# arr5 = np.array(data)
# print(arr5)
# print(arr5.shape)
# print(type(arr5))
# print(arr5.dtype)

# arr6 = np.zeros(10)
# print(arr6)
# print(arr6.shape)
# print(type(arr6))
# print(arr6.dtype)
#
# arr7 = np.zeros((3, 5))
# print(arr7)
# print(arr7.shape)
# print(type(arr7))
# print(arr7.dtype)
#
# arr8 = np.ones((4, 6))
# print(arr8)
# print(arr8.shape)
# print(type(arr8))
# print(arr8.dtype)
#
# arr9 = np.empty((2, 3, 2))
# print(arr9)
# print(arr9.shape)
# print(type(arr9))
# print(arr9.dtype)

arr10 = np.arange(10)
print(arr10)
print(arr10.shape)
print(type(arr10))
print(arr10.dtype)

print(arr10[5:8])   # 获取数组的第6个元素
arr10[5:8] = 10
print(arr10)

arr10_slice = arr10[5:8].copy()
arr10_slice[:] = 15
print(arr10_slice)
print(arr10)