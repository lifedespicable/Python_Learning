import re

def find_item(hero):
    with open('sanguo.txt',encoding='utf-8') as f:
        data = f.read().replace('\n', '')
        name_num = len(re.findall(hero,data))
        # print('主角 %s 出现了 %d 次'%(hero,name_num))
    return name_num

# 读取人物的信息
name_dict = {}
with open('name.txt') as f:
    for line in f:
        names = line.split('|')
        for n in names:
            # print(n)
            name_num = find_item(n)
            name_dict[n] = name_num

# 读取武器信息
weapon_dict = {}
weapon_list = []
i = 1
with open('weapon.txt',encoding='utf-8') as f:
    for line in f.readlines():
        if i%2 == 1:
            weapon_list.append(line.strip('\n'))
        i +=1
    for n in weapon_list:
        weapon_num = find_item(n)
        weapon_dict[n] = weapon_num

name_sorted = sorted(name_dict.items(),key=lambda item:item[1],reverse=True)
print(name_sorted[0:10])

weapon_sorted = sorted(weapon_dict.items(),key=lambda item:item[1],reverse=True)
print(weapon_sorted[0:10])