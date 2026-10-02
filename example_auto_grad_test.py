import micrograd_cpp as mcpp
import random

w = mcpp.Tensor(random.random())
x = mcpp.Tensor(1.0)
b = mcpp.Tensor(1.0)

y = w*x+b

print(y)

#you can also print data and grad of y
y.backward()
print(f"data: {y.data()} grad: {y.grad()}")

print(y+7)

#you can also do this
