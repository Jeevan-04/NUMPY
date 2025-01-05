import numpy as np

# From a list
a = np.array([1, 2, 3, 4])
print(a)

# From a tuple
b = np.array((5, 6, 7, 8))
print(b)

# Creating arrays with placeholder content
c = np.empty((2, 3))
print(c)

d = np.zeros((2, 3))
print(d)

e = np.ones((2, 3))
print(e)

f = np.full((2, 3), 7)
print(f)

# Creating arrays with ranges
g = np.arange(10, 20, 2)
print(g)

h = np.linspace(0, 1, 5)
print(h)

i = np.logspace(1, 3, 4)
print(i)

# Creating identity matrices
j = np.eye(3)
print(j)

k = np.identity(3)
print(k)

# Creating arrays from existing data
l = np.asarray([1, 2, 3])
print(l)

m = np.frombuffer(b'hello world', dtype='S1')
print(m)

n = np.fromfunction(lambda i, j: i + j, (3, 3))
print(n)

o = np.fromiter(range(5), dtype=int)
print(o)

p = np.fromstring('1 2 3 4 5', dtype=int, sep=' ')
print(p)

# Creating arrays with meshgrid
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
xx, yy = np.meshgrid(x, y)
print(xx)
print(yy)
