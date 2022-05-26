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
