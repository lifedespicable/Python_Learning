import time
# print(time.time())
# time.sleep(5)

# def i_can_sleep():
#     time.sleep(3)
#
# start_time = time.time()
#
# i_can_sleep()
#
# end_time = time.time()
#
# print('程序运行了 %s 秒'%(end_time-start_time))

# 此段代码是装饰器的演示demo

def timer(func):
    def wrapper():
        start_time = time.time()
        func()
        end_time = time.time()
        print(' 程序运行了 %s 秒' %( end_time - start_time))
    return wrapper

@timer
def i_can_sleep():
    time.sleep(3)

# 定义了函数之后一定要使用，才能发挥函数的效果
i_can_sleep()

# 这段代码的执行过程：
# timer(i_can_sleep())

timer(i_can_sleep())
