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
