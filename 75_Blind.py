class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        mymap = {}
        
        for i, n in enumerate(nums):
            diff = target - n
            if diff in mymap:
                return [mymap[diff], i]
            mymap[n] = i
    def maxProfit(self, prices: List[int]) -> int:
        minprof = float("inf")
        maxprof = 0
        
        for i in prices:
            if i < minprof:
                minprof = i
            elif i - minprof > maxprof:
                maxprof = i - minprof
            
        return maxprof          
    def containsDuplicate(self, nums: List[int]) -> bool:
        hashset = set();
        for i in nums:
            if(i in hashset):
                return True
            hashset.add(i)
        return False
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
    def maxSubArray(self, nums: List[int]) -> int:
        maxSub = nums[0]
        currSum = 0
        
        for n in nums:
            if currSum < 0 :
                currSum = 0
            currSum += n
            maxSub = max(maxSub, currSum)
        return maxSub
    def maxProduct(self, nums: List[int]) -> int:
        res = max(nums)
        currMax, currMin = 1,1 
        
        for n in nums:
            if n == 0:
                currMax, currMin = 1,1 
                continue
            temp = n * currMax
            currMax = max( n * currMax, n * currMin, n)
            currMin = min( temp, n * currMin, n)
            res = max(res, currMax)
        return res
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0 , len(nums) - 1
        
        while l <= r :
            mid = (l+r) // 2
            
            if nums[mid] == target:
                return mid
            
            if nums[mid] >= nums[l]:
                if nums[mid] < target or target < nums[l] :
                    l = mid + 1
                else:
                    r = mid - 1
            else:
                if nums[mid] > target or target > nums[r] :
                    r = mid - 1
                else:
                    l = mid + 1
        return -1
    def findMin(self, nums: List[int]) -> int:
        res = nums[0]
        l, r = 0, len(nums)-1
        
        while l <= r:
            if nums[l] <= nums[r]:
                res = min(res, nums[l])
            
            m = (l + r)  // 2
            res = min(res, nums[m])
            if nums[m] >= nums[r]:
                l = m + 1
            else:
                r = m - 1
        return res
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        
        for i, n in enumerate(nums):
            if i > 0 and n == nums[i-1]:
                continue
                
            l, r = i + 1, len(nums) - 1
            
            while l < r:
                threesum = n + nums[l] + nums[r]
                
                if threesum > 0:
                    r -= 1
                elif threesum < 0:
                    l += 1
                else:
                    res.append([n, nums[l], nums[r]])
                    l += 1
                    while nums[l] == nums[l -1] and l < r:
                        l += 1
        return res
    def maxArea(self, height: List[int]) -> int:
        l, r = 0 ,len(height) - 1
        res = 0
        
        while l <= r:
            area = (r - l) * min(height[l], height[r])
            res = max(res, area)
            
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return res
    def getSum(self, a: int, b: int) -> int:
        # 32 bits integer max
        MAX = 0x7FFFFFFF
        # 32 bits interger min
        MIN = 0x80000000
        # mask to get last 32 bits
        mask = 0xFFFFFFFF
        while b != 0:
            # ^ get different bits and & gets double 1s, << moves carry
            a, b = (a ^ b) & mask, ((a & b) << 1) & mask
        # if a is negative, get a's 32 bits complement positive first
        # then get 32-bit positive's Python complement negative
        return a if a <= MAX else ~(a ^ mask)   
