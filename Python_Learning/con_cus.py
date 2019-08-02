# 经典的生产者和消费者问题
from threading import Thread,current_thread
import time
import random    # 随机产生
from queue import Queue   # 实现队列

queue = Queue(5)     # 定义队列的长度

class ProducerThread(Thread):
    def run(self):
        name = current_thread().getName()
        nums = range(100)
        global queue
        while True:
            num = random.choice(nums)
            queue.put(num)
            print('生产者 %s 生产了 %s 数据' %(name, num))
            t = random.randint(1,3)
            time.sleep(t)
            print('生产者 %s 睡眠了 %s 秒' %(name, t))

class ConsumerThread(Thread):
    def run(self):
        name = current_thread().getName()
        global queue
        while True:
            num = queue.get()
            queue.task_done()
            print('消费者 %s 消耗了 %s 数据' %(name, num))
            t = random.randint(1,5)
            time.sleep(t)
            print('消费者 %s 睡眠了 %s 秒' %(name, t))

p1 = ProducerThread(name= 'p1')
p1.start()
p2 = ProducerThread(name= 'p2')
p2.start()
p3 = ProducerThread(name= 'p3')
p3.start()
c1 = ConsumerThread(name= 'c1')
c1.start()
c2 = ConsumerThread(name= 'c2')
c2.start()




