from collections import defaultdict
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        ans =0 
        count =0
        for i in nums:

            if i == 1:
                count +=1
            else:
                ans = max(ans, count)
                count = 0
        ans = max(ans, count)
        return ans
    def sortedSquares(self, nums: List[int]) -> List[int]:
        sq = []
        for i in range(len(nums)):
            sq.append(nums[i]*nums[i])
            sq.sort()
        return sq
    def findNumbers(self, nums: List[int]) -> int:
        count=0
        for i in nums:
            digit = math.floor(math.log10(i)+1)                
            if( digit % 2 == 0):
                count +=1
        return count   
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        i=0
        
        while(i<len(arr)):
            if(arr[i] != 0):
                i+=1
            else:
                arr.insert(i+1, 0)
                i+=2
                arr.pop()
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        for i in range(n):
            nums1.append(nums2[i])
            nums1.remove(0) 
        nums1.sort()
    def removeElement(self, nums: List[int], val: int) -> int:
        for i in range(nums.count(val)):
            nums.remove(val)
        return len(nums)                
    def removeDuplicates(self, nums: List[int]) -> int:
        i=0
        for l in range(1, len(nums)):
            if nums[l] == nums[i]:
                continue
            nums[i+1], nums[l] = nums[l], nums[i+1]
            i = i+1
        return i+1
    def checkIfExist(self, arr: List[int]) -> bool:
        count = 0
        for i in arr:
            if i == 0:
                count += 1
            if (i != 0) & (i * 2 in arr):
                return True
        return count > 1
    def validMountainArray(self, arr: List[int]) -> bool:
        n = len(arr)
        climbed = peakReached = False
        
        for i in range(1,n):
            prevNum, num = arr[i-1], arr[i]
            if num == prevNum: return False
            elif not peakReached:
                if num > prevNum: climbed = True
                elif prevNum > num: peakReached = True
            else:
                if num > prevNum: return False
        
        return climbed and peakReached
    def replaceElements(self, arr: List[int]) -> List[int]:
        for i in range(len(arr)-1):
            arr[i] = max(arr[i+1:len(arr)])
        
        arr.pop()
        arr.append(-1)
        return arr
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in nums:
            if i == 0:
                nums.remove(i)
                nums.append(i)
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        i = 0
        j = len(nums)-1
        while i<j:
            if nums[i]%2 != 0 and nums[j]%2 == 0:
                nums[i], nums[j] = nums[j], nums[i]
                i+=1
                j-=1
            elif nums[i]%2 == 0:
                i+=1
            elif nums[j]%2 != 0:
                j-=1
        return nums
    def heightChecker(self, heights: List[int]) -> int:
        temp = []
        
        # A copy of heights list
        for i in heights:
            temp.append(i)
        
        # Sorting heights list
        heights.sort()
        
        count = 0
        for i in range(len(heights)):
            if temp[i] != heights[i]:
                count += 1
        
        return count
    def thirdMax(self, nums: List[int]) -> int:
        nums = set(nums)
        if len(nums) < 3:
            return max(nums)
        
        return sorted(list(nums))[-3]  
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        dic = {}
        n = len(nums)
        
        for i in range(n):
            dic[nums[i]] = nums[i]
        lst = []  
        
        for i in range(1, n+1):
            if i not in dic:
                lst.append(i)
        return lst
 ### Starting Arrays and Strings chapter in Leetcode 
    def pivotIndex(self, nums: List[int]) -> int:
        s1,s2=0,sum(nums)
        for i in range(len(nums)):
            s2-=nums[i]
            if s1==s2:
                return i
            s1+=nums[i]
        return -1
    def dominantIndex(self, nums: List[int]) -> int:
        m=max(nums)
        a=nums.index(m)
        arr=[]
        g=0
        for i in nums:
            if i!=m:
                arr.append(i)
        if len(arr)!=0:
            g = max(arr)
        if m >= 2*g:
            return a
        return -1
    def plusOne(self, digits: List[int]) -> List[int]:
        s = ""
        for i in digits:
            s += str(i)
        
        s = int(s) + 1
        
        a = []
        for i in str(s):
            a.append(int(i))
        return (a)
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        if not mat or not mat[0]:
            return []
        m = len(mat)
        n = len(mat[0])
        diagonal_values = [[] for _ in range(m+n-1)]
        res = []
        for i in range(m):
            for j in range(n):
                diagonal_values[i + j].append(mat[i][j])
        
        res = diagonal_values[0]
        for i in range(1,len(diagonal_values)):
            if i % 2 == 0:
                res.extend(diagonal_values[i][::-1])
                
            else:
                res.extend(diagonal_values[i])
        return res
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res =[]
        left,right = 0, len(matrix[0])
        top, bottom = 0,len(matrix)
        
        while left<right and top<bottom:
            
            #for top :
            for i in range(left,right):
                res.append(matrix[top][i])
            top+=1
            
            #for right :
            for i in range(top,bottom):
                res.append(matrix[i][right-1])
            right-=1
            
            if not (left<right and top<bottom):
                break
            
            #for bottom :
            for i in range(right-1,left-1, -1):
                res.append(matrix[bottom-1][i])
            bottom-=1
            
            #for left :
            for i in range(bottom-1,top-1,-1):
                res.append(matrix[i][left])
            left+=1
            
        return res
# Pascal Triangle
    def generate(self, numRows: int) -> List[List[int]]:
        res=[[1]]
        for i in range(numRows - 1):
            temp = [0] + res[-1] + [0]
            row=[]
            for j in range(len(res[-1]) + 1):
                row.append(temp[j] + temp[j+1])
            res.append(row)
        return res
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        lptr, total = 0,0
        res = 99999
        for rptr in range(len(nums)):
            total += nums[rptr]
            while total >= target:
                res = min(rptr - lptr + 1, res)
                total -= nums[lptr]
                lptr += 1
        
        return 0 if res == 99999 else  res
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        sums = 0
        for i in range(0,len(nums), 2):
            sums += nums[i]
        
        return sums
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        res =[]
        ptr1 = 0
        ptr2 = len(numbers) - 1
        
        while ptr1 < ptr2:
            sums = numbers[ptr1] + numbers[ptr2]
            
            if sums > target:
                ptr2 -=1
            elif sums < target:
                ptr1 +=1
            else:
                return [ptr1 + 1, ptr2 + 1]
