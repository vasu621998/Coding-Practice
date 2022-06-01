#Leetcode solutions for String questions
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        res = ""
        carry = 0
        
        a,b = a[::-1], b[::-1]
        
        for i in range(max(len(a),len(b))):
            #ord used to fetch ASCII value
            digitA= ord(a[i]) - ord("0") if i < len(a) else 0
            digitB= ord(b[i]) - ord("0") if i < len(b) else 0
            
            total = digitA + digitB +carry
            temp = str(total%2)
            res = temp + res
            carry = total // 2
        
        if carry:
            res = "1" + res
            
        return res
    def strStr(self, haystack: str, needle: str) -> int:
        
        if needle in haystack:
            return haystack.index(needle)
        else:
            return -1
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res=""
        for i in range(len(strs[0])):
            for s in strs:
                if i==len(s) or s[i] != strs[0][i]:
                    return res
            
            res += strs[0][i]
        return res
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        #return s.reverse() ##Reverse method
        #s[:] = s[-1:-1] ## Slicing method
        ##Using two pointers:
        
        #first pointer
        ptr1 = 0
        #Second pointer 
        ptr2 = len(s) - 1
        
        while ptr1 < ptr2:
            s[ptr1], s[ptr2] = s[ptr2], s[ptr1]
            ptr1 += 1
            ptr2 -= 1
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        sums = 0
        for i in range(0,len(nums), 2):
            sums += nums[i]
        
        return sums
