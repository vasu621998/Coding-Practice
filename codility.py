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

#Odd Occurrence Array
def Solution(A):
  A.sort()
  A.append(-1)
  for i in range(0, len(A), 2):
    if A[i] != A[i+1]:
      return A[i]
   
# Frog Jump
def Solution(X, Y, D):
  v = (Y - X) // D
  if X + v*D >= Y:
    return v
  else:
    return v+1
 
   
# Perm Missing Element
def Solution(A):
  if len(A) == 0:
    return 1
  A.sort()
  for i in range(0, len(A)):
    if A[i] != i+1:
      return i+1
   return (len(A)+1)

# tape equilibrium
def Solution(A):
  if len(A) < 2:
    return 0
  s = Sum(A)
  minDiff = 2000
  k = 0
  for i in range(0, len(A)-1):
    k += A[i]
    diff = abs(2*k - s)
    minDiff = min(minDiff, diff)
   return minDiff


# Frog River One
def solution(X, A):
    leaves = {}
    for second in range(0, len(A)):
        leaves[A[second]] = True
        condition = len(leaves)
        if len(leaves) == X:
          return second         
    return -1

# Max counters
def solution(N, A):
    R = [0] * N 
    m = 0
    b = 0
    for i in range(0, len(A)):
      if A[i] <= N :
        R[A[i] - 1] = max(b, R[A[i] - 1]) + 1
        m = max(m, R[A[i] - 1)
             
      else:
         b = m
                     
    for i in range(0, len(R)):
        if (R[i] < b):
                     
           R[i] = b   
         
# Missing Positive integer       
def solution(A):
    A.sort()
    if(A[len(A) - 1] <= 0):
        return 0
    iso = false
    for i in range(0, len(A)):
        if A[i] == 1:
           iso = true
    if iso == false:
        return 1
                     
    for i in range(0, len(A - 1)):
         if A[i] > 0 and (A[i+1] - A[i]) > 1:
            return A[i] + 1
    return A[len(A) - 1] + 1

# Count Divisibles
def Solution(A,B,K):
  c = int(B/K) - int(A/K)
  if(A%K == 0):
        c = c+1
  return c
                     
                   
# Genomic Range Query
def solution(S, P, Q):
    # write your code in Python 3.6
    ans = []
    for (p, q) in zip(P,Q):
        if 'A' in S[p:q+1]:
            ans.append(1)
        elif 'C' in S[p:q+1]:
            ans.append(2)
        elif 'G' in S[p:q+1]:
            ans.append(3)
        else:
            ans.append(4)
            
    return ans
