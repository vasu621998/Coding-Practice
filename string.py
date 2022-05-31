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
