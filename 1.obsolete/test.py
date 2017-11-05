import numpy
A = numpy.array([[1, 0, 1],
                 [2, 0, 1],
                 [3, 0, 0],
                 [4, 0, 0],
                 [5, 0, 0]])
# A = numpy.array([[1,2,3],[4,5,6]])
# A1 = A[A[:, 2] == 1, :] # extract all rows with the third column 1
# A0 = A[A[:, 2] == 0, :] # extract all rows with the third column 0
# A1 = [0 if a > 1 for a in A]
A[A != 1] = 0
print(A)
# print(A1)
