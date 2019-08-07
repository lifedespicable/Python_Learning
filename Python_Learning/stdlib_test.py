# 日常应用比较广泛的模块是：
# 1、文字处理的：re
# 2、日期类型的time、datetime
# 3、数字和数学类型的math和random
# 4、文件和目录访问的pathlib、os.path
# 5、数据压缩和归档的tarfile
# 6、通用操作系统的os、logging、argparse
# 7、多线程的threading、queue
# 8、Internet数据处理的base64、json、urllib
# 9、结构化标记处理工具的html、xml
# 10、开发工具的unitest
# 11、测试工具的timeit
# 12、软件包发布的venv
# 13、运行服务的_main_

# 正则表达式中的元字符
# '.'：匹配任意的单个字符；'^'：表示以什么样的内容作为开头；'$'：表示以什么样的内容作结尾，表示从后面向前面去匹配
# '*'：匹配前面的这一个字符出现0次或者多次；'+'：匹配前面的这一个字符出现1次或者多次
# '?'：匹配前面的这一个字符出现0次或者1次；'{m}'：表示前面的这一个字符必须要出现m次
# '{m,n}'：表示前面的这一个字符必须要出现m次~n次，'[]'表示中括号里面的任意一个字符匹配成功就可以匹配成功了
# '|'：表示我们的字符选择左边或者是右边，通常会和我们后面的括号用在一起
# '\d'：表示我们匹配的内容是一串数字；'\d'是相当于[0-9]+
# '\D'：表示我们匹配不包含数字的；'\s'：表示匹配的是一个字符串
# '()'：表示的是要进行分组
# '^$'：表示的是这一行是空行
# '.*?'：表示的是不使用贪婪模式，只匹配我们第一个匹配上的内容；'*'匹配会尽可能长的去匹配

import re
# p = re.compile('a')
# print(p.match('a'))
# print(p.match('b'))

# p = re.compile('ca*t')
# print(p.match('cat'))
# print(p.match('caaaaat'))
# print(p.match('ct'))

# p = re.compile('...')
# print(p.match('a'))
# print(p.match('d'))

p = re.compile('jpg$')  # 表示去匹配以'jpg'为结尾的字符串

# 判断一个文件是否是以jpg结尾
# filename = 'test.jpg'
# print(filename.endswith('jpg'))

# p = re.compile('ca{4,6}t')
# print(p.match('cat'))
# print(p.match('caaaat'))
# print(p.match('caaaaat'))
# print(p.match('caaaaaat'))
# print(p.match('caaaaaaat'))
# print(p.match('ct'))

# p = re.compile('c[bcd]t')
# print(p.match('cat'))
# print(p.match('cbt'))
# print(p.match('cct'))
# print(p.match('cdt'))
# print(p.match('cet'))
# print(p.match('cft'))

# p = re.compile('.{3}')
# print(p.match('bat'))

# p = re.compile('....-..-..')
# print(p.match('2018-05-16'))

# p = re.compile(r'(\d+)-(\d+)-(\d+)')
# print(p.match('aa2018-05-16bb').group(3))
# print(p.match('2018-05-16').groups())
# year, month, day = p.match('2018-03-18').groups()
# print(year)
# print(month)
# print(day)

# print(p.search('aa2018-05-16bb'))

# print('\nx\n')      # 如果不加'r'的话它是显示出来一个换行符，'r'就是表示告诉Python后面的内容原样输出，不要进行转义
# print(r'\nx\n')

# phone = '123-456-789 # 这是电话号码'
# p2 = re.sub(r'#.*$','',phone)
# print(p2)
# p3 = re.sub(r'\D','',p2)
# print(p3)

import random
# print(random.randint(1,5))
# print(random.choice(['aa','bb','cc','dd','ee']))