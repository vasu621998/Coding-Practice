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
        
