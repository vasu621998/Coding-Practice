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
