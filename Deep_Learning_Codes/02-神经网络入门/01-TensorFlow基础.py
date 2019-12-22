import theano.tensor as T
from theano import function

x = T.dscalar('x')
y = T.dscalar('y')
b = T.constant(1)
z = x * y + b
f = function([x, y], z)
print(f(2, 3))
print(type(f(2, 3)))