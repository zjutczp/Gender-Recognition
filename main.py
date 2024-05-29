import numpy as np
import math


sum = 0
for t in range(20000):
    rows = 10
    cols = 3
    mean = 0
    std_dev = 1
    matrix1 = np.random.randint(0,70,size=(rows, cols))
    for i in range(rows):
        for j in range(cols):
            matrix1[i][j] = 70 * matrix1[i][j] / np.sum(matrix1)

    matrix2 = np.random.randint(0,70,size=(rows, cols))
    for i in range(rows):
        for j in range(cols):
            matrix2[i][j] = 70 * matrix2[i][j] / np.sum(matrix2)

    matrix = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            matrix[i][j] = abs(matrix1[i][j] - matrix2[i][j])
    # print(matrix1)
    # print("---------------")
    # print(matrix2)
    # print("---------------")
    # print(matrix)
    P = 1- np.sum(matrix)/2
    sum += P

mean = sum/20000
print(mean)
