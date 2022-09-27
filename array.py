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
    def majorityElement(self, nums: List[int]) -> int:
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        return max(count, key= count.get)
# T: O(N)
# S: O(N)    
    def combinationSum4(self, nums: List[int], target: int) -> int:
        @lru_cache(None)
        def helper(target):
            if target == 0: return 1
            elif target < 0: return 0
            else: return sum([helper(target - num) for num in nums])
            
        return helper(target)       
    
#Input: nums = [1,2,3], target = 4
#Output: 7
#Explanation:
#The possible combination ways are:
#(1, 1, 1, 1)
#(1, 1, 2)
#(1, 2, 1)
#(1, 3)
#(2, 1, 1)
#(2, 2)
#(3, 1)
#Note that different sequences are counted as different combinations.
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        low = 0
        high = len(nums)-1
        left,right = 0,0
        while low <= high:
			# find midpoint
            mid = low + ( high - low ) // 2
            
			# THE ONLY IMPORTANT PART
			# When midpoint matches the target we'll find the range
            if nums[mid] == target:
                left, right = mid,mid
				
				# Can we go left?
                if nums[mid-1] == target: 
                    left = mid - 1
				
				# Can we go right and not out of bounds?
                if mid + 1 < len(nums) and nums[mid+1] == target: 
                    right = mid + 1
				
				# mid is at index 0 so we don't have any left
                if left < 0: 
                    left = 0
					
				 # mid is at last index so we don't have any right
                if right >= len(nums) - 1:
                    right = len(nums) - 1
				
				# If we can go left, keep going left
                while left > 0 and nums[left-1] == target: 
                    left -= 1
					
				# If we can go right, keep going right
                while right < len(nums)-1 and nums[right+1] == target: #
                    right += 1
                    
                return [left,right]
            
			# Adjust bound (just like any other Binary Search problem)
            if target > nums[mid]:
                low = mid + 1
            else:
                high = mid - 1
                
        return [-1,-1]   
    
#Input: nums = [5,7,7,8,8,10], target = 8
#Output: [3,4]    

# T: O(logN)
# S: O(1)
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        
        path = []
        answer = []
        def dp(idx, total):
            if total == target:
                answer.append(path[:])
                return
            if total > target:
                return
            
            for i in range(idx, len(candidates)):
                path.append(candidates[i])
                dp(i, total + candidates[i])
                path.pop()
        
        dp(0, 0)
        return answer
# Input: candidates = [2,3,6,7], target = 7
#Output: [[2,2,3],[7]]
#Explanation:
#2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
#7 is a candidate, and 7 = 7.
#These are the only two combinations.
# T: O(N)
# S: O(1)
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        if len(nums) < 4: return []
        
        nums.sort()
        res = []
        
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                l = j+1
                r = len(nums)-1
                while l < r:

                    sum_ = nums[i]+nums[j]+nums[l]+nums[r]
                    a = [nums[i], nums[j], nums[l], nums[r]]
                
                    if sum_ == target and a not in res:
                        res.append(a)
                  
                    if sum_ > target:
                        r -= 1
                    
                    else:
                        l += 1
                        while l < r and nums[l-1] == nums[l]:
                            l += 1
                  
        return res 
    def minSetSize(self, arr: List[int]) -> int:
        d = {}
        for x in arr:
            if x not in d:
                d[x] = 1
            else:
                d[x] += 1
                  
        l = sorted(d.values())
        N = len(arr) // 2
        idx = 0
        
        while N > 0:
            N -= l[-idx-1]
            idx += 1
            
        return idx 
	def isPossible(self, nums):
		count = Counter(nums)
		numLen = len(nums)

		lastSub = defaultdict(int)

		for num in nums:
			
			if not count[num]:
				continue

			count[num] -= 1

			# append to existing sub list
			if lastSub[num - 1]:
				lastSub[num - 1] -= 1
				lastSub[num] += 1

			# add new sub list
			elif count.get(num + 1) and count.get(num + 2):
				count[num + 1] -= 1
				count[num + 2] -= 1
				lastSub[num + 2] += 1

			else:
				return False

		return True
    def isPowerOfFour(self, n: int) -> bool:
        if n%4 != 0 and n!=1:
            return False

        division = n
        while division >= 4:
            division = division/4

        return division == 1
    def reorderedPowerOf2(self, n: int) -> bool:
        n1 = sorted(str(n))
    
        for i in range(30):
            res = sorted(str(2 ** i))
            if res == n1:
                return True
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        l, r = 0, len(matrix) - 1
        
        while l < r:
            for i in range(r - l):
                top, bottom = l, r
                
                topLeft = matrix[top][l + i]
                
                matrix[top][l + i] = matrix[bottom - i][l]
                
                matrix[bottom - i][l] = matrix[bottom][r - i]
                
                matrix[bottom][r - i] = matrix[top + i][r]
                
                matrix[top + i][r] = topLeft
                
            r -= 1
            l += 1
# T: O(N^2)
# S: O(1)
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        output = []  # This is where the permutations shall get appended
        def recurse(num, count, taken):  # A backtracking recursive function that will do the job
            """
            num: our temporary list for the current permutation being constructed
            count: length of the current permutation
            taken: list indicating whether or not a specific element has been included in the current permutation
            """
            
            # Base case: Case when we have created a permutation of the length we want after recursion 
            if count == len(nums):  
                if num not in output:
                    output.append(num)
                return
            
            # Otherwise, pick an element from the remaining elements and place it at the end of current list
            # Perform this operation recursively to obtain all possible permutations
            else:
                for i in range(len(nums)):
                    if taken[i] is not True:
                        recurse(num+[nums[i]], count+1, taken[:i]+[True]+taken[i+1:])
        
        # Call the recursive function starting from an empty permutation with no count and nothing taken at first
        recurse([], 0, [False]*len(nums))
        
        # The above function modifies the global output list which contains our desired output
        return output
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        m, n = len(matrix), len(matrix[0])
        res = -inf
        
        for l in range(n):
            rowSums = [0] * m
            for r in range(l, n):
                colSums = [0]
                colSum = 0
                for i in range(m):
                    rowSums[i] += matrix[i][r]
                    colSum += rowSums[i]
                    diff = colSum - k
                    idx = bisect_left(colSums, diff)
                    if idx < len(colSums):
                        if colSums[idx] == diff:
                            return k
                        else:
                            res = max(res, colSum - colSums[idx])
                    insort(colSums, colSum)
        return res
    def numIslands(self, grid: List[List[str]]) -> int:
      m = len(grid)
      n = len(grid[0])
      
      def traverseLand(i,j):
        if(i>=0 and i<m and j>=0 and j<n and grid[i][j]=='1'):
          grid[i][j]='0'
          traverseLand(i+1,j)
          traverseLand(i,j+1)
          traverseLand(i-1,j)
          traverseLand(i,j-1)
        
      op=0
      for i in range(m):
        for j in range(n):
          if(grid[i][j]=='1'):
            op+=1
            traverseLand(i,j)
      
      return op 
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        n = len(nums)
        dp = [1 for i in range(n)]
        hashh = [i for i in range(n)]
        ans_ind = 0
        
        for i in range(1, n):
            for j in range(0,i):
                if nums[i]%nums[j] == 0 and dp[j]+1 > dp[i]:            
                    dp[i] = dp[j]+1
                    hashh[i] = j
                    
                    # print(dp)
                    # print(hashh)
        out = []
        maxi = dp[0]
        
        for i in range(len(nums)):
            if dp[i] > maxi:
                ans_ind = i
                maxi = dp[i]
        
        while(hashh[ans_ind]!=ans_ind):
            out.append(nums[ans_ind])
            ans_ind = hashh[ans_ind]
        out.append(nums[ans_ind])
        return(out)
    def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
        seen = dict()
        count = 0
        for x in nums:
            if x - diff in seen and x - 2 * diff in seen:
                count += 1
            seen[x] = True
        return count
# T: O(N)
# S: O(N)

#Input: nums = [0,1,4,6,7,10], diff = 3
#Output: 2
#Explanation:
#(1, 2, 4) is an arithmetic triplet because both 7 - 4 == 3 and 4 - 1 == 3.
#(2, 4, 5) is an arithmetic triplet because both 10 - 7 == 3 and 7 - 4 == 3. 
        result = []
        # Use backtracking to gather all possible results
        def backtracking(current_list):
            nonlocal n, result
            # If length already matches, then record this result
            if len(current_list) == n:
                result.append("".join(current_list))
                return
            # Get the last digit and calculate the next valid digit
            last_digit = int(current_list[-1])
            # Positive difference, goto next layer of backtracking if next digit is valid
            if last_digit + k < 10:
                current_list.append(str(last_digit + k))
                backtracking(current_list)
                current_list.pop()
            # Negative difference, notice that we put k != 0 here to avoid getting duplicate answers
            if last_digit - k >= 0 and k != 0:
                current_list.append(str(last_digit - k))
                backtracking(current_list)
                current_list.pop()        
        # Try all leading digits from [1-9], notice that leading 0 is invalid
        for i in range(1,10):
            backtracking([str(i)])
        return result 
    def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
        tokens.sort()
        n = len(tokens)
        i, j = 0, n
        while i < j:
            if tokens[i] <= power:
                power -= tokens[i]
                i += 1
            elif i - (n - j) and j > i + 1:
                j -= 1
                power += tokens[j]
            else: break
        return i - (n - j)        
    
# T: O(NlogN)
# S: O(1)

#Input: tokens = [100], power = 50
#Output: 0
#Explanation: Playing the only token in the bag is impossible because you either have too little power or too little score.
    def trap(self, height: List[int]) -> int:
        length = len(height)
        left = maxi = 0
        right = length - 1
        left_max = right_max = 0
        ans = 0
        while left <= right:
            if left_max <= right_max:
                if left_max < height[left]:
                    left_max = height[left]
                else:
                    ans += left_max - height[left]
                left += 1
            else:
                if right_max < height[right]:
                    right_max = height[right]
                else:
                    ans += right_max - height[right]
                right -= 1

        return ans       

# T: O(N)
# S: O(1)
    def firstMissingPositive(self, nums: List[int]) -> int:
        nums = sorted(list(set(nums)))
        i = 1
        for num in nums:
            if num == i:
                i += 1
            elif num > 0:
                return i
        return i
# T: O(N)
# S: O(1)
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        strnum2 = ''.join([chr(x) for x in nums2])
        strmax = ''
        ans = 0
        for num in nums1:
            strmax += chr(num)
            if strmax in strnum2:
                ans = max(ans,len(strmax))
            else:
                strmax = strmax[1:]
        return ans 
    def solveNQueens(self, n: int) -> List[List[str]]:
        colset = set()
        negdiag = set()
        posdiag = set()
        board = [["."] * n for i in range(n)]
        res = []
        
        def backtracking(r):
            if r ==  n :
                copy = ["".join(row) for row in board]
                res.append(copy)
                return 
            for c in range(n):
                if c in colset or (r+ c) in posdiag or (r - c) in negdiag:
                    continue
                    
                colset.add(c)
                negdiag.add(r -c)
                posdiag.add(r + c)
                board[r][c] = "Q"
                
                backtracking(r + 1)
                colset.remove(c)
                negdiag.remove(r-c)
                posdiag.remove(r + c)
                board[r][c] = "."
        backtracking(0)
        return res 
    def canJump(self, nums: List[int]) -> bool:
        flag=True
        n = len(nums)
        for i in range(n):
            if nums[i]==0 and i!=n-1:
                flag=False
                for j in range(i-1,-1,-1):
                    if nums[j]>=i-j+1:
                        flag=True
                        break
            if not flag:
                return False
        return flag 
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        size = len(intervals)

        # find the insert position using binary search
        L = 0
        R = size - 1
        while(L <= R):
            mid = (L + R) >> 1
            if (intervals[mid][0] <= newInterval[0]):
                L = mid + 1
            else:
                R = mid - 1

        # insert the new interval
        if (L == size):
            intervals.append(newInterval)
        else:
            intervals.insert(L, newInterval)
        i = max(R, 0)
        size += 1

        # if necessary, merge intervals
        while(i + 1 < size):
            while(i + 1 < size) and (intervals[i][1] >= intervals[i + 1][0]):
                other = intervals.pop(i + 1)
                intervals[i][0] = min(intervals[i][0], other[0])
                intervals[i][1] = max(intervals[i][1], other[1])
                size -= 1
            i += 1

        return intervals
    
# T: O(N)
# S: O(1)

# Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
# Output: [[1,2],[3,10],[12,16]]
# Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
    def generateMatrix(self, n: int) -> List[List[int]]:
        r1 = 0
        r2 = n -1 
        c1 = 0
        c2 = n -1
        
        res = [[0 for i in range(n)] for i in range(n)]
        print(res)
        k = 1
        while(r1<=r2 and c1 <= c2):
            for i in range(r1,r2+1):
                res[r1][i] = k
                k+=1

            for i in range(c1+1,c2+1):
                res[i][c2]=k
                k+=1

            for i in range(r2-1,r1-1,-1):
                res[c2][i] = k
                k+=1

            for i in range(c2-1,c1,-1):
                res[i][c1] = k
                k+=1

            r1+=1
            c1+=1
            c2-=1
            r2-=1
            
        return res
    
# T: O(N)
# S: O(1)

# Input: n = 3
# Output: [[1,2,3],[8,9,4],[7,6,5]]
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        cur = preSum = sum([num for num in nums if num % 2 == 0])
        res = []
        
        for i, pair in enumerate(queries):
            index = pair[1]
            value = pair[0]
            if nums[index] % 2 == 0:
                cur += value if value % 2 == 0 else -nums[index]
            else:
                cur += nums[index] + value if value % 2 != 0 else 0        
            nums[index] += value
            res.append(cur)
        
        return res  
        rows = len(grid)
        cols = len(grid[0])
        dp = [[0]*cols for i in range(0, rows)]
        dp[0][0] = grid[0][0]
        
        for i in range(0, rows):
            for j in range(0, cols):
                if i-1 >= 0 and j-1 >= 0:
                    top = dp[i-1][j]
                    left = dp[i][j-1]
                    dp[i][j] = min(top + grid[i][j], left + grid[i][j])
                elif i-1 >= 0:
                    top = dp[i-1][j]
                    dp[i][j] = top + grid[i][j]
                elif j-1 >= 0:
                    left = dp[i][j-1]
                    dp[i][j] = left + grid[i][j]
                    
        return dp[rows-1][cols-1]
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if root is None:
            return
        
        ans = []
        def dfs(node,count,path):
            
            if not node.left and not node.right:
                if count+node.val == targetSum:
                    path.append(node.val)
                    ans.append(path[:])
                    path.pop()
                    
                return

            path.append(node.val)

            if node.left :
                dfs(node.left,count+node.val,path)
            if node.right:
                dfs(node.right,count+node.val,path)
                
            path.pop()
            return
        
        
        dfs(root,0,[])
        return list(ans) 
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        
        subset = []
        
        def dfs(i):
            if i >= len(nums):
                res.append(subset.copy())
                return
            subset.append(nums[i])
            dfs(i + 1)
            
            subset.pop()
            dfs(i + 1)
            
        dfs(0)
        return res
    
# T : O(N * 2^N)
# S : O(1)

# Input: nums = [1,2,3]
# Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        l, r = 0, len(nums) - 1
        i = 0
        
        def swap(i, j):
            tmp = nums[i]
            nums[i] = nums[j]
            nums[j] = tmp
        
        while i <= r:
            if nums[i] == 0:
                swap(l,i)
                l += 1
            
            elif nums[i] == 2:
                swap(i,r)
                r -= 1
                i -= 1
            i += 1
# T : O(N)
# S : O(1)

# Input: nums = [2,0,2,1,1,0]
# Output: [0,0,1,1,2,2]
    def minDistance(self, word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)
        dp = [[0 for _ in range(m+1)] for _ in range(n+1)]
        
        for i in range(1,n+1):
            dp[i][0] = i

        for j in range(1,m+1):
            dp[0][j] = j
            
            
        for i in range(1,n+1):
            for j in range(1,m+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    
                else:
                    dp[i][j] = 1+ min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
                    
        return dp[-1][-1]        
    
# T: O(N)
# S: O(1)

# Input: word1 = "horse", word2 = "ros"
# Output: 3
# Explanation: 
# horse -> rorse (replace 'h' with 'r')
# rorse -> rose (remove 'r')
# rose -> ros (remove 'e')
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        m = len(matrix[0])
        col = True
        
        for i in range(n):
            if matrix[i][0] == 0:
                col = False
            for j in range(1, m):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0  
					
        for i in range(n-1,-1,-1):
            for j in range(1, m):
                print(i,j)
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
            if col == False:
                matrix[i][0] = 0
                
# S: O(1)

# Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
# Output: [[1,0,1],[0,0,0],[1,0,1]]
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        low, high = 0, len(matrix)-1
        while low <= high:
            mid = (low+high)//2
            if target < matrix[mid][0]:
                high = mid-1
            elif target > matrix[mid][0]:
                low = mid+1
            else:
                return True
        
        row = high
        l, r = 0, len(matrix[0])-1
        while l <= r:
            mid = (l+r)//2
            if target < matrix[row][mid]:
                r = mid-1
            elif target > matrix[row][mid]:
                l = mid+1
            else:
                return True
        
        return False


# T: O(log m + log n)    
# S: O(1)

# Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
# Output: true
    def equationsPossible(self, equations: List[str]) -> bool:
        equals: dict[str, list[str]] = defaultdict(list)
        non_equals: list[str] = []

        for eq in equations:
            if eq[1] == "!":
                if eq[0] == eq[3]:
                    return False
                
                non_equals.append(eq)
                continue
            
            equals[eq[0]].append(eq[3])
            equals[eq[3]].append(eq[0])
        
        var_to_group: dict[str, set[str]] = {}

        for var in equals.keys():
            if var in var_to_group:
                continue
            group: set[str] = set()
            links: list[str] = [var]

            while links:
                link = links.pop()
                
                if link in group:
                    continue
                group.add(link)
                var_to_group[link] = group
                links.extend(equals[link])
        
        for ne in non_equals:
            if ne[0] in var_to_group.get(ne[3], set()):
                return False
        
        return True
        
# T : O(N)
# S : O(1)

# Input: equations = ["a==b","b!=a"]
# Output: false
# Explanation: If we assign say, a = 1 and b = 1, then the first equation is #satisfied, but not the second.
#There is no way to assign the variables to satisfy both equations.
    def removeDuplicates(self, nums: List[int]) -> int:
        l = 0
        for r in range(len(nums)):
            if nums[l] == nums[r]:
                if r - l + 1 > 2:
                    nums[r] = None
            else:
                l = r

        current_index = 0
        for i in range(len(nums)):
            nums[current_index] = nums[i]

            if nums[i] is not None:
                current_index += 1

        return current_index
    
# T : O(N)
# S : O(1)

# Input: nums = [1,1,1,2,2,3]
# Output: 5, nums = [1,1,2,2,3,_]
# Explanation: Your function should return k = 5, with the first five elements of nums # being 1, 1, 2, 2 and 3 respectively.
# It does not matter what you leave beyond the returned k (hence they are underscores).
