import numpy as np

arr1 = [[1,2,3,4],
       [5,6,7,8]]

arr2 = [[9,8,7,6],
        [5,4,3,2]]

grid = np.meshgrid(range(4), range(4))
print(grid)

grid = np.stack(grid, axis=-1)
print(grid[:,:,0])
print(grid[:,:,1])

