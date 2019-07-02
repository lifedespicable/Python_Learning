# print('abc',end='\n')
# print('abc')

# def func (a,b,c):
#     print('a = %s ' % a)
#     print('b = %s ' % b)
#     print('c = %s ' % c)

# func(1,c= 3)

# 取得参数的个数
# def howlong (first,*other):
#     print(1 + len(other))
#
# howlong()

# var1 = 123
# def func ():
#     global var1
#     var1= 456
#     print(var1)
#
# func()
# print(var1)

# iter() next()

# list1 = [1,2,3]
# it = iter(list1)
# print(next(it))
# print(next(it))
# print(next(it))
# print(next(it)) # except

# for i in range(10,20,0.5):
#     print(i)

# def frange (start,end,step):
#     x = start
#     while x < end :
#         yield (x)
#         x += step

# for i in frange(10,20,0.5):
#     print(i)

# def true ():return True
#
# true()
#
# lambda : True
#
# def add (a,b):return a+b
#
# print(add(3,5))
#
# lambda x,y : x+y
#
# lambda x: x <= (month, day),
#
# def fun1 (x):
#     return x <= (month, day)
#
# lambda item:item[1]
#
# def fun2 (item):
#     return item[1]
#
# adict = {'a':'aa','b':'bb'}
# for i in adict.items():
#     print(fun2(i))

# filter()
# map()
# reduce()
# zip()

# a = [1,2,3,4,5,6,7]
# print(len(list(filter(lambda x:x >2, a))))
#
# a = [1,2,3]
# b = [2,3,4]
# print(list(map(lambda x:x + 1, b)))
# print(map(lambda x:x, b))
# print(list(map(lambda x,y:x+y,a,b)))

# from functools import reduce
# print(reduce(lambda x,y:x+y,[2,3,4],1))

# 实现元祖的对调
# print(list(zip((1,2,3),(4,5,6))))
# for i in zip((1,2,3),(4,5,6)):
#     print(i)

# 实现字典的键值对的对调
# dicta = {'a':'aa','b':'bb'}
# dictb = zip(dicta.values(),dicta.keys())
# print(dict(dictb))

def func():
    a = 1
    b = 2
    return a + b

def sum (a):
    def add (b):
        return a + b
    return add

# add 函数的名称或者函数的引用
# add() 函数的调用

num1 = func()
num2 = sum(2)
print(type(num1))
print(type(num2))
print(num2(4))
