def fib(n):
    a, b = 1, 1
    for i in range(1, n):
        a, b = b, a + b
    return a

print(fib(1))
print(fib(2))
print(fib(3))
print(fib(4))
print(fib(5))
print(fib(6))
print(fib(7))
print(fib(8))

print('---------')

fib_2 = lambda x: 1 if x <= 2 else fib_2(x - 1) + fib_2(x - 2)
print(fib_2(1))
print(fib_2(2))
print(fib_2(3))
print(fib_2(4))
print(fib_2(5))
print(fib_2(6))
print(fib_2(7))
print(fib_2(8))
