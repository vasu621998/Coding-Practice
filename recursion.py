class Solution:
    def fib(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        
        return self.fib(n-1) + self.fib(n-2)
    def climbStairs(self, n: int) -> int:
        one, two = 1,1
        
        for i in range(n-1):
            temp = one
            one = one + two
            two = temp
            
        return one
