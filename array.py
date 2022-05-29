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
        m = len(mat)
        n = len(mat[0])
        diagonal_values = defaultdict(list)
        res = []
        for i in range(m):
            for j in range(n):
                diagonal_values[i + j].append(mat[i][j])
        
        for i, values in diagonal_values.items():
            if i % 2 == 0:
                res.extend(values[::-1])
                
            else:
                res.extend(values)
        return res
