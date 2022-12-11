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
    def myPow(self, x: float, n: int) -> float:
        def helper(x,n):
            if x == 0:
                return 0
            if n == 1:
                return x
            
            res = helper(x , n // 2)
            res = res * res
            
            return x * res if  n % 2 else res
            
        res = helper(x, abs(n))
        
        return res if n >= 0 else 1 / res
    def myPow(self, x: float, n: int) -> float:
        if n < 0:
            x = 1 / x
            n = -n
        
        return self.helper(x, n)
    
    def helper(self, x, n):
        if n == 0:
            return 1
        
        partialAnswer = self.helper( x, n // 2 )
        if n % 2 == 1:
            partialAnswer = partialAnswer * partialAnswer * x
        else:
            partialAnswer = partialAnswer * partialAnswer
        
        return partialAnswer
    def kthGrammar(self, n: int, k: int) -> int:
        if n == 1 and k == 1:
            return 0
        
        mid = pow(2, n - 1) // 2
        
        if k <= mid:
            return self.kthGrammar(n-1,k)
            
        else:
            return (self.kthGrammar(n-1,k-mid)^1)            
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        def build(l, r):
            trees = []
            if l > r:
                trees.append(None)                  # when 1 > 0, 1 as a root has no children
                                                    # when l == r, it is a leaf node   
            for n in range(l, r + 1):
                                                    # left and right are unique as we only traverse each range once
                left = build(l, n - 1)  
                right = build(n + 1, r)
                
                for _l in left:                     # left and right are presented as arrays
                    for _r in right:
                        root = TreeNode(n)          # build root based on different combo of left and right
                        root.left = _l
                        root.right = _r
                        trees.append(root)          # append the unique tree
            return trees
        
        return build(1, n)   
    def isPerfectSquare(self, num: int) -> bool:
        if int((num**.5))**2 == num:
            return True
        return False
