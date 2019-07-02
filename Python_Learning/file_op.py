# 将小说的主要人物记录在文件中
file1 = open('name.txt', 'w')
file1.write('诸葛亮')
file1.close()
#
# file2 = open('name.txt')
# print(file2.read())
# file2.close()
#
# file3 = open('name.txt','a')
# file3.write('刘备')
# file3.close()

# file4 = open('name.txt')
# print(file4.readline())
#
# file5 = open('name.txt')
# for line in file5.readlines():
#     print(line)
#     print('======')

file6 = open('name.txt')
print('当前文件的指针位置是' +str(file6.tell()))
print('当前读取到了一个字符，字符的内容是 %s' %file6.read(1))
print('当前文件的指针位置是' +str(file6.tell()))
# seek的第一个参数是表示的是偏移位置，其实是表示seek函数的相对位置，第二个参数 0表示的是从文件开头开始偏移，
# 1表示从当前位置开始偏移，2表示从文件结尾开始偏移
file6.seek(5,0)
print('我们进行了seek的操作')
print('当前文件的指针位置是' +str(file6.tell()))
print('当前读取到了一个字符，字符的内容是 %s' %file6.read(1))
print('当前文件的指针位置是' +str(file6.tell()))
file6.close()
