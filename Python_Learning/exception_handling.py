# i = j

# print())

# a = '123'
# print(a[3])

# d = {'a':1 ,'b':2}
# print(d['c'])

# year = int(input('请输入年份：'))

# try:
#     year = int(input('请输入年份：'))
# except ValueError:
#     print('年份要输入数字')

# a = 123
# a.append()

# try:
#     a.append()
# except AttributeError:
#     print('属性错误，该数据类型不具备该属性')

# try:
#     year = int(input('请输入年份：'))
# except (ValueError,AttributeError,KeyError):
#     print('该异常属于值异常或者是属性异常或者是键异常')
# try:
#     print(1/'a')
# except Exception as c:
#     print('%s'%c)

# try:
#     raise NameError('HelloError')
# except NameError:
#     print('My Custom Error')

try:
    a = open('name.txt')
except Exception as c:
    print('%s'%c)

finally:
    a.close()




