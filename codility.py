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
