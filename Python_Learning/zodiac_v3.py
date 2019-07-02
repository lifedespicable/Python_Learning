chinese_zodiac = '猴鸡狗猪鼠牛虎兔龙蛇马羊'
zodiac_name = (u'摩羯座', u'水瓶座', u'双鱼座', u'白羊座', u'金牛座', u'双子座', u'巨蟹座',
               u'狮子座', u'处女座', u'天秤座', u'天蝎座', u'射手座')
# 下面是一个元组的嵌套
zodiac_days = ((1, 20), (2, 19), (3, 21), (4, 21), (5, 21), (6, 22), (7, 23), (8, 23),
               (9, 23), (10, 23), (11, 23), (12, 23))
# (month, day) = (2, 15)

cz_num = {}
for i in chinese_zodiac:
    cz_num[i] = 0

z_num = {}
for i in zodiac_name:
    z_num[i] = 0

# zodiac_day = filter(lambda x: x <= (month, day), zodiac_days)
# # print(zodiac_day)
# # print(list(zodiac_day))
# zodiac_len = len(list(zodiac_day)) % 12
# # print(len(list(zodiac_day)))
# print(zodiac_name[zodiac_len])
# i = 0
    # for zodiac_day in zodiac_days :
    #     if zodiac_day <= (int_month,int_day) :
    #         i = i+1
    # print('%i 月 %i 日的星座是' % (int_month,int_day)+zodiac_name[i])

    # for zd_num in range(len(zodiac_days)):
    #     if zodiac_days[zd_num] >= (int_month,int_day):
    #         print(zodiac_name[zd_num])
    #         break
    #     elif int_month == 12 and int_day > 23:
    #         print(zodiac_name[0])
    #         break
while True:
    # 用户输入出生年份月份和日期
    year = int(input('请用户输入出生年份：'))
    int_month = int(input('请输入出生月份：'))
    int_day = int(input('请输入出生日期：'))

    n = 0
    while zodiac_days[n] < (int_month,int_day):
     if int_month == 12 and int_day > 23:
        break
     n += 1
    # 输出生肖和星座
    print(zodiac_name[n])

    print('%s 年的生肖是 %s' % (year,chinese_zodiac[year % 12]))

    cz_num[chinese_zodiac[year % 12]] += 1
    z_num[zodiac_name[n]] += 1

    #输出生肖和星座的打印信息
    for each_key in cz_num.keys():
        print('生肖 %s 有 %d 个'%(each_key,cz_num[each_key] ))

    for each_key in z_num.keys():
        print('%s 有 %d 个'%(each_key,z_num[each_key]))


