# Algorithm

## CSC稀疏矩阵压缩

### 定义

在矩阵中，若数值为0的元素数目远远多于非0元素的数目，并且非0元素分布没有规律时，则称该矩阵为**稀疏矩阵**；与之相反，若非0元素数目占大多数时，则称该矩阵为**稠密矩阵**。

### 描述

**csc_matrix**按列压缩CSC—Compressed sparse column。

```python
>>> import numpy as np
>>> from scipy.sparse import csc_matrix
>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
array([[1, 0, 4],
       [0, 0, 5],
       [2, 3, 6]])
# 按col列来压缩
# 对于第i列，非0数据行是indices[indptr[i]:indptr[i+1]] 数据是data[indptr[i]:indptr[i+1]]
# 在本例中
# 第0列，有非0的数据行是indices[indptr[0]:indptr[1]] = indices[0:2] = [0,2]
# 数据是data[indptr[0]:indptr[1]] = data[0:2] = [1,2],所以在第0列第0行是1，第2行是2
# 第1行，有非0的数据行是indices[indptr[1]:indptr[2]] = indices[2:3] = [2]
# 数据是data[indptr[1]:indptr[2] = data[2:3] = [3],所以在第1列第2行是3
# 第2行，有非0的数据行是indices[indptr[2]:indptr[3]] = indices[3:6] = [0,1,2]
# 数据是data[indptr[2]:indptr[3]] = data[3:6] = [4,5,6],所以在第2列第0行是4，第1行是5,第2行是6
```

### 用处

eyeriss中用于压缩常量数据。EAAC工程中用到。

### 实现

todo