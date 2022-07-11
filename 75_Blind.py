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
