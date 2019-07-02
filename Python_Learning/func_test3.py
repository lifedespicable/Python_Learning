# a*x+b=y
def a_line (a,b):
    # def arg_y (x):
    #     y = a*x+b
    #     return y
    y = lambda x:a*x+b
    return y

# a=3,b=5
# x=10,y=?
# x=20,y=?

line1 = a_line(3,5)
line2 = a_line(5,10)
print(line1(10))
print(line1(20))
print(line2(10))
print(line2(20))