import numpy as np

# np.take()
a = np.array([4, 3, 5, 7, 6, 8])
indices = [0, 1, 4]
print("np.take:", np.take(a, indices))  # [4 3 6]

# np.put()
np.put(a, [0, 2], [-44, -55])
print("np.put:", a)  # [-44 3 -55 7 6 8]

# np.choose()
choices = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
print("np.choose:", np.choose([2, 0, 1, 2], choices))  # [ 9  2  7 12]

# Advanced Indexing
# np.ix_()
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
ixgrid = np.ix_(a, b)
print("np.ix_:", ixgrid)

# np.r_[]
print("np.r_:", np.r_[1:4, 0, 4])  # [1 2 3 0 4]

# np.c_[]
print("np.c_:", np.c_[1:4, 0:3])  # [[1 0] [2 1] [3 2]]

# Boolean Indexing
# np.where()
a = np.array([1, 2, 3, 4])
print("np.where:", np.where(a > 2))  # (array([2, 3]),)

# np.nonzero()
print("np.nonzero:", np.nonzero(a > 2))  # (array([2, 3]),)

# np.extract()
print("np.extract:", np.extract(a > 2, a))  # [3 4]

# Masking
# np.putmask()
a = np.array([1, 2, 3, 4])
np.putmask(a, a > 2, [42, 43, 44, 45])
print("np.putmask:", a)  # [ 1  2 42 43]

# np.ma.masked_where()
a = np.array([1, 2, 3, 4])
print("np.ma.masked_where:", np.ma.masked_where(a > 2, a))  # [1 2 -- --]

# np.ma.masked_equal()
print("np.ma.masked_equal:", np.ma.masked_equal(a, 2))  # [1 -- 3 4]

# np.ma.masked_greater()
print("np.ma.masked_greater:", np.ma.masked_greater(a, 2))  # [1 2 -- --]

# np.ma.masked_less()
print("np.ma.masked_less:", np.ma.masked_less(a, 2))  # [-- 2 3 4]

# np.ma.masked_inside()
print("np.ma.masked_inside:", np.ma.masked_inside(a, 2, 3))  # [1 -- -- 4]

# np.ma.masked_outside()
print("np.ma.masked_outside:", np.ma.masked_outside(a, 2, 3))  # [-- 2 3 --]

# Fancy Indexing
# np.ogrid[]
print("np.ogrid:", np.ogrid[0:5, 0:5])

# np.mgrid[]
print("np.mgrid:", np.mgrid[0:5, 0:5])

# Special Indexing
# np.flat
a = np.array([[1, 2], [3, 4]])
print("np.flat:", a.flat[1])  # 2

# np.ndindex()
print("np.ndindex:", list(np.ndindex(a.shape)))  # [(0, 0), (0, 1), (1, 0), (1, 1)]

# np.ndenumerate()
print("np.ndenumerate:", list(np.ndenumerate(a)))  # [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)]

# Index Modification
# np.delete()
a = np.array([1, 2, 3, 4])
print("np.delete:", np.delete(a, [1, 3]))  # [1 3]

# np.insert()
print("np.insert:", np.insert(a, 1, 5))  # [1 5 2 3 4]

# np.append()
print("np.append:", np.append(a, [5, 6]))  # [1 2 3 4 5 6]

# Slicing Utilities
# np.s_[]
print("np.s_:", a[np.s_[1:3]])  # [2 3]

# np.index_exp[]
print("np.index_exp:", a[np.index_exp[1:3]])  # [2 3]

# np.take_along_axis()
a = np.array([[10, 30, 20], [60, 40, 50]])
indices = np.array([[0, 1, 2], [2, 0, 1]])
print("np.take_along_axis:", np.take_along_axis(a, indices, axis=1))  # [[10 30 20] [50 60 40]]

# np.put_along_axis()
a = np.array([[10, 30, 20], [60, 40, 50]])
np.put_along_axis(a, indices, [1, 2, 3, 4, 5, 6], axis=1)
print("np.put_along_axis:", a)  # [[ 1  2  3] [ 4  5  6]]

# np.unravel_index()
print("np.unravel_index:", np.unravel_index([22, 41, 37], (7, 6)))  # (array([3, 6, 6]), array([4, 5, 1]))

# np.ravel_multi_index()
print("np.ravel_multi_index:", np.ravel_multi_index(([3, 6, 6], [4, 5, 1]), (7, 6)))  # [22 41 37]

# np.isin()
a = np.array([1, 2, 3, 4])
print("np.isin:", np.isin(a, [2, 3]))  # [False  True  True False]

# np.ma.getmask()
a = np.ma.array([1, 2, 3], mask=[0, 1, 0])
print("np.ma.getmask:", np.ma.getmask(a))  # [False  True False]

# np.ma.mask_or()
a = np.ma.array([1, 2, 3], mask=[0, 1, 0])
b = np.ma.array([4, 5, 6], mask=[1, 0, 0])
print("np.ma.mask_or:", np.ma.mask_or(a, b))  # [ True  True False]

# np.ma.mask_and()
print("np.ma.mask_and:", np.ma.mask_and(a, b))  # [False False False]

# np.ma.nomask
print("np.ma.nomask:", np.ma.nomask)  # False

# np.diag_indices()
print("np.diag_indices:", np.diag_indices(3))  # (array([0, 1, 2]), array([0, 1, 2]))

# np.triu_indices()
print("np.triu_indices:", np.triu_indices(3))  # (array([0, 0, 0, 1, 1, 2]), array([0, 1, 2, 1, 2, 2]))

# np.tril_indices()
print("np.tril_indices:", np.tril_indices(3))  # (array([0, 1, 1, 2, 2, 2]), array([0, 0, 1, 0, 1, 2]))

# np.lib.stride_tricks.as_strided()
a = np.array([1, 2, 3, 4, 5, 6])
print("np.lib.stride_tricks.as_strided:", np.lib.stride_tricks.as_strided(a, shape=(3, 2), strides=(8, 8)))

# np.meshgrid()
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print("np.meshgrid:", np.meshgrid(x, y))

# np.broadcast_to()
a = np.array([1, 2, 3])
print("np.broadcast_to:", np.broadcast_to(a, (3, 3)))

# np.squeeze()
a = np.array([[[0], [1], [2]]])
print("np.squeeze:", np.squeeze(a))  # [0 1 2]

# np.roll()
a = np.array([1, 2, 3, 4, 5])
print("np.roll:", np.roll(a, 2))  # [4 5 1 2 3]

# np.rollaxis()
a = np.ones((3, 4, 5))
print("np.rollaxis:", np.rollaxis(a, 2, 0).shape)  # (5, 3, 4)

# np.swapaxes()
a = np.array([[1, 2, 3]])
print("np.swapaxes:", np.swapaxes(a, 0, 1))  # [[1] [2] [3]]

# np.moveaxis()
a = np.ones((3, 4, 5))
print("np.moveaxis:", np.moveaxis(a, 0, -1).shape)  # (4, 5, 3)

# np.transpose()
a = np.array([[1, 2, 3]])
print("np.transpose:", np.transpose(a))  # [[1] [2] [3]]

# np.arange()
print("np.arange:", np.arange(3))  # [0 1 2]

# np.linspace()
print("np.linspace:", np.linspace(2.0, 3.0, num=5))  # [2.   2.25 2.5  2.75 3.  ]

# np.logspace()
print("np.logspace:", np.logspace(2.0, 3.0, num=4))  # [ 100.  215.443469  464.158883 1000. ]

# np.eye()
print("np.eye:", np.eye(3))  # [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]

# np.identity()
print("np.identity:", np.identity(3))  # [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]

# np.diag()
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("np.diag:", np.diag(a))  # [1 5 9]