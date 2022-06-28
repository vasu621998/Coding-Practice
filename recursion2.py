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
                
        
