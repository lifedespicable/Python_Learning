# 下面这段代码的意思是如果不传入FIRST的值，那么就将FIRST的值设为0
# 如果传入FIRST的值话，就传入FIRST的值


def counter(FIRST=0):
    cnt = [FIRST]

    def add_one():
        cnt[0] += 1
        return cnt[0]
    return add_one


num5 = counter(5)
num10 = counter(10)

print(num5())
print(num5())
print(num5())
print(num10())
print(num10())
print(num10())


# def counter_my ():
#     i = 0
#     def add_one_my():
#         i += 1
#         return i
#     return add_one_my
#
# num1 = counter_my()
# print(num1)
# print(num1())
# print(num1())
# print(num1())
# print(num1())
# print(num1())
# print(num1())
# print(num1())
