# user1 = {'name':'Tom','HP':'100'}
# user2 = {'name':'Jerry','HP':'80'}
#
# def print_role(rolename):
#     print('Name is %s , HP is %s' %(rolename['name'],rolename['HP']))
#
# print_role(user1)
# print_role(user2)

class Player():     # 定义一个类
    def __init__(self, Name , HP, Occu):
        self.__name = Name       # 在面向对象编程中，变量被称作是属性，而函数被称作是方法
        self.hp = HP
        self.occu = Occu
    def print_role(self):      # 定义一个方法
        print('Name is %s , HP is %s , Occupation is %s' %(self.__name, self.hp, self.occu))

    def updateName(self,newname):
        self.__name = newname

class Monster():
    '定义怪物类'
    def __init__(self, hp = 100):
        self.HP = hp

    def run(self):
        print('移动到某个位置')

    def whoami(self):
        print('我是怪物父类')

class Animals(Monster):
    '普通怪物'
    def __init__(self, hp = 10):
        # self.HP = hp
        super().__init__(hp)    # 在告诉Python程序，Animals中不用再次初始化hp的值了，因为这在Monster里面已经初始化了

class Boss(Monster):
    'Boss类怪物'
    def __init__(self, hp = 1000):
        super().__init__(hp)
    def whoami(self):
        print('我是怪物我怕谁')

# user3 = Player('Tom', 110, 'warrior')      # 类的实例化
# user4 = Player('Jerry', 90 , 'master')
#
# user3.print_role()
# # user4.print_role()
#
# user3.updateName('Wilson')
# user3.print_role()
# # user4.print_role()
#
# user3.__name = 'aaa'
# user3.print_role()

a1 = Monster(200)
print(a1.HP)
print(a1.run())

a2 = Animals()
print(a2.HP)
print(a2.run())

a3 = Boss(800)
print(a3.HP)
a3.whoami()

print('a1的类型是 %s' %(type(a1)))
print('a2的类型是 %s' %(type(a2)))
print('a3的类型是 %s' %(type(a3)))

print(isinstance(a2, Monster))
print(isinstance(a3, Monster))
print(isinstance(a1, Monster))

# cat = Animals(100)
# print(cat.HP)
# cat.run()