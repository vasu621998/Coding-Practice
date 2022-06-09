# Binray Gap
def Solution(N):
  n = bin(N) [2:]
  b = 0
  count = 0
  
  for i in N:
    if int(i) == 0:
      b += 1
     elif int(i) == 1:
      count = max(b, count)
      b = 0
  return count

# Cyclic Array
def Solution(A, K):
  N = len(A)
  B = [None] * N
  count = 0
  for i in range(0,N):
    B[(i+K)%N] = A[i]
  return B

