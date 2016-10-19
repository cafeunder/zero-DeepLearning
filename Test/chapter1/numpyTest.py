import numpy as np

A = np.array([[1,2], [3,4]])

print(A)
print(A.shape)  #(2,2)
print(A.dtype)  #int32

print(A[A%2 == 0])  #[2,4]
