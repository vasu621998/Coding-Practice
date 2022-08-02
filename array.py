import math
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
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        lptr, rptr = 0 , len(nums) - 1
        while lptr < rptr:
            nums[lptr], nums[rptr] = nums[rptr], nums[lptr]
            lptr += 1
            rptr -= 1
        
        lptr, rptr = 0 , k - 1
        while lptr < rptr:
            nums[lptr], nums[rptr] = nums[rptr], nums[lptr]
            lptr += 1
            rptr -= 1
            
        lptr, rptr = k , len(nums) - 1
        while lptr < rptr:
            nums[lptr], nums[rptr] = nums[rptr], nums[lptr]
            lptr += 1
            rptr -= 1
    def getRow(self, rowIndex: int) -> List[int]:
        res = [1]*(rowIndex + 1)
        for i in range(1, rowIndex):
            for j in range(i,0,-1):
                res[j] += res[j-1]
        return res
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix = 1
        res = [1] * n
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]
        
        postfix = 1        
        for i in range(len(nums)-1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]
        return res
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        l = []
        while matrix:
            l.extend(matrix.pop(0))
            
        l = sorted(l)
        
        low = 0
        high = len(l)
        while low < high:
            mid = (low + high) // 2
            
            if l[mid] == target:
                return True
            elif l[mid] < target:
                low = mid + 1
            else:
                high = mid
                
        return False
    def maxProfit(self, prices: List[int]) -> int:
        #buying. = i + 1
        # selling = i + 2
        
        dp = {} # key = (i, buying), val = profit
        
        def dfs(i, buying):
            if i >= len(prices):
                return 0
            
            if (i,buying) in dp:
                return dp[(i, buying)]
            
            if buying:
                buy = dfs(i+1, not buying) - prices[i]
                cooldown = dfs(i+1, buying)
                dp[(i,buying)] = max(buy,cooldown)
            else:
                sell = dfs(i+2, not buying) + prices[i]
                cooldown = dfs(i+1, buying)
                dp[(i,buying)] = max(sell,cooldown)
            return dp[(i, buying)]
        
        return dfs(0, True)
    def checkPossibility(self, nums: List[int]) -> bool:
        changed = False
        
        for i in range(len(nums) - 1):
            if nums[i] <= nums[i + 1]:
                continue
            
            if changed:
                return False
            
            if i == 0 or nums[i + 1] >= nums[i - 1]:
                nums[i] = nums[i+1]
            else:
                nums[i+1] = nums[i]
            changed = True
        return True
    def runningSum(self, nums: List[int]) -> List[int]:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        return nums
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        # ptr = 0
        # res = []
        
        # b = len(nums) // 2
        
        # for i in range(b):
            # res.append(nums[i])
            # res.append(nums[n])
            # n += 1
        # return res
        i = 0
        out = []
        while i < len(nums)/ 2:
            out.append(nums[i])
            out.append(nums[i+n])
            i += 1
        return out
    def mySqrt(self, x: int) -> int:
        num = 0
        while True:
            if num * num > x:
                return num-1
                break
            num += 1
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        
        
        while l <= r:
            mid = (l + r)// 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return l
    def getConcatenation(self, nums: List[int]) -> List[int]:
        nums = nums * 2
        
        return nums
    def reverse(self, x: int) -> int:
        MAX = 2147483647
        MIN = - 2147483648
        
        res = 0
        while x:
            digit = int(math.fmod(x, 10))
            x = int(x/10)
            
            if res > (MAX// 10) or (res == MAX//10 and digit >= MAX%10):
                return 0
            if res < MIN// 10 or (res == MIN//10 and digit <= MIN%10):
                return 0
            
            res = (res * 10) + digit
        return res
    def largestNumber(self, nums: List[int]) -> str:
        for i, num in enumerate(nums):
            nums[i] = str(num)
            
            
        def compare(n1, n2):
            if n1 + n2 > n2 + n1:
                return -1
            else:
                return 1
            
        nums = sorted(nums, key = cmp_to_key(compare))
        
        
        return str(int("".join(nums)))
    def buildArray(self, nums: List[int]) -> List[int]:
        ans = []
        
        for i in range(len(nums)):
            ans.append(nums[nums[i]])
            
        return ans
    
# T: O(N)
# S: O(1)
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        max_health = 0
        for i in accounts:
            max_health = max(max_health, sum(i))
        
        return max_health

# Input: accounts = [[1,2,3],[3,2,1]]
# Output: 6
# Explanation:
# 1st customer has wealth = 1 + 2 + 3 = 6
# 2nd customer has wealth = 3 + 2 + 1 = 6
# Both customers are considered the richest with a wealth of 6 each, so return 6.
    
# T:O(N)
# S:O(1)
    def numIdenticalPairs(self, nums: List[int]) -> int:
        ans = 0
        
        
        seen = {}
        
        
        for i,n in enumerate(nums):
            
            if n in seen:
                ans+= seen[n]
            
            seen[n] = seen.get(n,0)+1
            
        return ans        

#Input: nums = [1,2,3,1,1,3]
#Output: 4
#Explanation: There are 4 good pairs (0,3), (0,4), (3,4), (2,5) 0-indexed.    
    
# T: O(N)
# S: O(N)
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        max_candy = max(candies)
        res = []
        
        for candy in candies:
            
            if candy + extraCandies >= max_candy :
                res.append(True)
            else:
                res.append(False)
            
        return res
    
# Input: candies = [2,3,5,1,3], extraCandies = 3
# Output: [true,true,true,false,true] 
# Explanation: If you give all extraCandies to:
# - Kid 1, they will have 2 + 3 = 5 candies, which is the greatest among the kids.
# - Kid 2, they will have 3 + 3 = 6 candies, which is the greatest among the kids.
# - Kid 3, they will have 5 + 3 = 8 candies, which is the greatest among the kids.
# - Kid 4, they will have 1 + 3 = 4 candies, which is not the greatest among the kids.
# - Kid 5, they will have 3 + 3 = 6 candies, which is the greatest among the kids.    

# T: O(N) + O(N)
# S: O(N)
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        temp = nums[:]
        
        temp.sort()
        
        return [temp.index(n) for n in nums]
    
# Input: nums = [8,1,2,2,3]
# Output: [4,0,1,1,3]
# Explanation: 
# For nums[0]=8 there exist four smaller numbers than it (1, 2, 2 and 3). 
# For nums[1]=1 does not exist any smaller number than it.
# For nums[2]=2 there exist one smaller number than it (1). 
# For nums[3]=2 there exist one smaller number than it (1). 
# For nums[4]=3 there exist three smaller numbers than it (1, 2 and 2).
    
# T:O(N)
# S:O(N)
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        for ind, r in enumerate(image):
            image[ind] = r[ : : -1]

        for r in image:
            for ind, num in enumerate(r):
                if (num == 0):
                    r[ind] = 1
                else:
                    r[ind] = 0

        return image
    
    
# T: O(N^2)
# T: O(N)
    def isPowerOfTwo(self, n: int) -> bool:
        if n<=0:
            return 0
        return int(log2(n))==log2(n)
    
# Input: n = 1
# Output: true
# Explanation: 20 = 1
    def uniquePaths(self, m: int, n: int) -> int:
        row = [1] * n
        
        for i in range(m -1):
            newRow = [1] * n
            for j in range(n-2, -1, -1):
                newRow[j] = newRow[j+1] + row[j]
            
            row = newRow
        return row[0]

#Input: m = 3, n = 2
#Output: 3
#Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
#1. Right -> Down -> Down
#2. Down -> Down -> Right
#3. Down -> Right -> Down
    
#T: O(M+N)
#S: O(N)
    def divide(self, dividend: int, divisor: int) -> int:
        check = False
    
        if dividend < 0 or divisor < 0:
            check = True
        
        if dividend < 0 and divisor < 0:
            check = False
        
        
        res = abs(dividend) // abs(divisor)
    
    
    
        if check:
            res = -res
        return min(max(-2147483648,res),2147483647);
    
# Input: dividend = 7, divisor = -3
# Output: -2
# Explanation: 7/-3 = -2.33333.. which is truncated to -2.
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        temp = []
        
        for row in matrix:
            temp.extend(row)
        temp.sort()
        return temp[k-1]
    
#Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
#Output: 13
#Explanation: The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13

# T: O(NlogN * MlogM)
# S: O(N)
    def findLHS(self, nums: List[int]) -> int:
        res = 0
        counter = Counter(nums)
        for c in counter:
            if c+1 in counter:
                res = max(res, counter[c] + counter[c+1])
        return res
# Input: nums = [1,3,2,2,5,2,3,7]
# Output: 5
# Explanation: The longest harmonious subsequence is [3,2,2,2,3].
# T: O(N)
# S: O(N)
