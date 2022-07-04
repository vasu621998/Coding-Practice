class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def merge_sort(arr):

            if len(arr)<=1:
                return arr

            start = 0
            end = len(arr)
            mid = (start+end)//2

            left = arr[:mid]
            right = arr[mid:]

            merge_sort(left)
            merge_sort(right)

            i = j = k = 0


            while i < len(left) and j < len(right):

                if left[i] <= right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1

            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1

            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1
                
        merge_sort(nums)
        return nums
    def totalNQueens(self, n: int) -> int:
        col = set()
        posDiag = set() # r + c
        negDiag = set() # r - c
        
        
        res = 0
        def backtrack(r):
            if r == n:
                
                nonlocal res
                res += 1
                return
            
            for c in range(n):
                
                if c in col or (r+c) in posDiag or (r-c) in negDiag:
                    continue
                
                col.add(c)
                posDiag.add(r+c)
                negDiag.add(r-c)
                backtrack(r+1)                
                col.remove(c)
                posDiag.remove(r+c)
                negDiag.remove(r-c)
                
        backtrack(0)
        return res                    
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        
        def backtrack(start, comb):
            if len(comb) == k:
                res.append(comb.copy())
                return
        
            for i in range(start, n+1):
                comb.append(i)
                
                backtrack(i + 1, comb)
                
                comb.pop()
        
        backtrack(1, [])
        
        return res                
        
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        for i in range(9):
            for j in range(9):
                num = board[i][j]
                rows[i].add(num)
                cols[j].add(num)
                boxID = (i//3)*3+j//3
                boxes[boxID].add(num)

        def fill(r, c):
            if c == 9:
                if r == 8:
                    return True
                else:
                    c = 0
                    r += 1
            if board[r][c] == '.': # we should fill in number
                for num in range(1, 10): # choose the number from 1 to 9
                    num = str(num)
                    if num in rows[r] or num in cols[c] or num in boxes[r//3*3+c//3]: continue
                    # num is valid for current cell, we can try further
                    rows[r].add(num)
                    cols[c].add(num)
                    boxes[r//3*3+c//3].add(num)
                    board[r][c] = num
                    if fill(r, c+1):
                        return True
                    # fail to fill, need to backtrack
                    rows[r].remove(num)
                    cols[c].remove(num)
                    boxes[r//3*3+c//3].remove(num)
                    board[r][c] = '.'
                return False
            else:
                return fill(r, c+1)
            
        fill(0,0)
        
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    def generateParenthesis(self, n: int) -> List[str]:
        # Add Open parenthesis if openN < n
        # Add Closed parenthesis if closedN < openN
        # Valid if openN == closedN == n
        stack = []
        res = []
        
        def backtrack(openN, closedN):
            if openN == closedN == n:
                res.append("".join(stack))
                return
            
            if openN < n:
                stack.append("(")
                backtrack(openN + 1, closedN)
                stack.pop()
            
            if closedN < openN:
                stack.append(")")
                backtrack(openN, closedN + 1)
                stack.pop()
            
        backtrack(0,0)
        return res
    def largestRectangleArea(self, heights: List[int]) -> int:
        maxArea = 0
        stack = [] #pair index & height
        
        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                index,height = stack.pop()
                maxArea = max(maxArea, height * (i - index))
                start = index
            stack.append((start,h))
            
            
        for i,h in stack:
            maxArea = max(maxArea, h * (len(heights) - i))
        return maxArea
    def letterCombinations(self, digits: str) -> List[str]:
        res = []
        digitToChar = {
            "2" : "abc",
            "3" : "def",
            "4" : "ghi",
            "5" : "jkl",
            "6" : "mno",
            "7" : "pqrs",
            "8" : "tuv",
            "9" : "wxyz"
        }
        def backTracking(i, currentStr):
            if len(currentStr) == len(digits):
                res.append(currentStr)
                return
            for c in digitToChar[digits[i]]:
                backTracking(i+1, currentStr+c)
        if digits:
            backTracking(0, "")
        
        return res
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        res = []
        events = [] 
        q = [0]
        heapq.heapify(q)
        left = []
        right = []
        
        for b in buildings:
            left.append([b[0], -b[-1], "start"])
    
        #1. Sort start edges by height from high -> low
        left.sort(key = lambda x: x[1])
    
        for b in buildings:
            right.append([b[1], -b[-1], "end"])
        
        #1. Sort end edges by height from low -> high
        right.sort(key = lambda x: x[1], reverse=True)
        
        #1. Sort points by left edge
        events = left + right
        events.sort(key = lambda x: x[0])
         
        for e in events:
            #2.  Get the start edges
            if e[-1] == "start":
                # If the current edge is higher than the current highest edges,
                # add it to the queue 
                if -e[1] > -heapq.nsmallest(1, q)[0]:
                    res.append([e[0], -e[1]])
                heapq.heappush(q, e[1])
                
            #3.  Get the end edges
            if e[-1] == "end":
                # Remove the height of this edge from the queue
                q.remove(e[1])
                
                # If the height of this edge was the highest edges,
                # add the position of this edge with the remain highest edges to the answer
                if -e[1] > -heapq.nsmallest(1, q)[0]:
                    res.append([e[0],-heapq.nsmallest(1, q)[0]])
            
        return res      
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        temp = 0
        if len(nums) == 1:
            return [nums[:]]

        for _ in range(len(nums)):
            temp = nums.pop(0)
            p = self.permute(nums)
            for j in p:
                j.append(temp)
            res.extend(p)
            nums.append(temp)
        return res
