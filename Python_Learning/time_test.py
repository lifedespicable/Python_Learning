import time
import datetime

# print(time.time())
# print(time.localtime())
# print(time.strftime('%Y-%m-%d %H:%M:%S'))
# print(time.strftime('%y-%m-%d'))
# print(time.strftime('%Y-%M-%d'))
# print(time.strftime('%Y%m%d'))

print(datetime.datetime.now())
newtime = datetime.timedelta(minutes=10)
print(datetime.datetime.now() + newtime)

one_day = datetime.datetime(2018,8,3,20,13,56)
addtime = datetime.timedelta(days= 10)
print(one_day)
print(one_day + addtime)