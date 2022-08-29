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
    def reverseWords(self, s: str) -> str:
        return ' '.join(s.strip().split()[::-1])
    def reverseWords(self, s: str) -> str:
        return ' '.join(a[::-1] for a in s.split())
    def isSubsequence(self, s: str, t: str) -> bool:
        i,j = 0,0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
                
            j += 1
            
        return True if i == len(s) else False
    
# Input: s = "abc", t = "ahbgdc"
# Output: true            
# T : O(N)                
# S : O(1)
    def multiply(self, num1: str, num2: str) -> str:
        if "0" in [num1, num2] :
            return "0"
        
        res = [0] * (len(num1) + len(num2))
        
        num1, num2 = num1[::-1], num2[::-1]
        
        for i1 in range(len(num1)):
            for i2 in range(len(num2)):
                digit = int(num1[i1]) * int(num2[i2])
                res[i1+i2] += digit
                res[i1+i2 + 1] += (res[i1+i2] // 10)
                res[i1+i2] = res[i1+i2] % 10
            
        res, beg = res[::-1], 0
        
        while beg < len(res) and res[beg] == 0:
            beg += 1
            
        
        res = map(str, res[beg:])
        return "".join(res)
#Input: num1 = "123", num2 = "456"
#Output: "56088"    
    
# T: O(N * M)
# S: O(N + M)
    def toLowerCase(self, s: str) -> str:
        return s.lower()
    
# T :O(N)
# S: O(1)
    def defangIPaddr(self, address: str) -> str:
        res = ""
        
        for char in address:
            if char == ".":
                res = res + "[" + char + "]"
            else:
                res += char
        
        return res
    
# Input: address = "255.100.50.0"
# Output: "255[.]100[.]50[.]0"
# T: O(N)
# S : O(1)
    def finalValueAfterOperations(self, operations: List[str]) -> int:
        ans = 0
        
        for opr in operations:
            if "-" in opr:
                ans = ans - 1
            else:
                ans = ans + 1
                
        return ans

#Input: operations = ["--X","X++","X++"]
#Output: 1
#Explanation: The operations are performed as follows:
#Initially, X = 0.
#--X: X is decremented by 1, X =  0 - 1 = -1.
#X++: X is incremented by 1, X = -1 + 1 =  0.
#X++: X is incremented by 1, X =  0 + 1 =  1.    
    
# T: O(N)
# S: O(1)
    def mostWordsFound(self, sentences: List[str]) -> int:
        res = 0
        for sentence in sentences:
            
            res = max(res, len(sentence.split()))
        
        return res
#Input: sentences = ["please wait", "continue to fight", "continue to win"]
#Output: 3
#Explanation: It is possible that multiple sentences contain the same number of words. 
#In this example, the second and third sentences (underlined) have the same number of words.
    
# T: O(N)
# S: O(1)
    def interpret(self, command: str) -> str:
        res = ""
        for i in range(len(command)):
            if command[i] == "(" and command[i+1] ==")":
                res += "o"
            elif command[i] == "(" or command[i] == ")":
                continue
            elif command[i] == "G":
                res += "G"
            else:
                res += command[i]
                
        return res
    
# Input: command = "G()(al)"
# Output: "Goal"
# Explanation: The Goal Parser interprets the command as follows:
# G -> G
# () -> o
# (al) -> al
# The final concatenated result is "Goal".
    
# T : O(N)
# S : O(1)
    def restoreString(self, s: str, indices: List[int]) -> str:
        
        res = [" " for _ in range(len(indices))]
        
        for i in range(len(indices)):
            
            res[indices[i]] = s[i]
        
        return "".join(res)

# Input: s = "codeleet", indices = [4,5,6,7,0,2,1,3]
# Output: "leetcode"
# Explanation: As shown, "codeleet" becomes "leetcode" after shuffling.

# T : O(N)
# S : O(1)
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        string = ""
        step = (numRows - 1)*2
        down = 0 
        for i in range(numRows):
            if i < len(s):
                string += s[i]
            j = i
            while j < len(s):
                j += step
                if(step and j < len(s)): 
                    string += s[j]
                j += down
                if(down and j <len(s)):
                    string += s[j]
            step -= 2
            down += 2
        return string   
#Input: s = "PAYPALISHIRING", numRows = 4
#Output: "PINALSIGYAHRPI"
#Explanation:
#P     I    N
#A   L S  I G
#Y A   H R
#P     I
    def balancedStringSplit(self, s: str) -> int:
        ans = 0
        r_count = l_count = 0
        for char in s:
            if char == 'R':
                r_count += 1
            elif char == 'L':
                l_count += 1
            if r_count == l_count:
                ans += 1
                r_count = l_count = 0
            continue
        return ans

# Input: s = "RLRRLLRLRL"
# Output: 4
# Explanation: s can be split into "RL", "RRLL", "RL", "RL", each substring contains same number of 'L' and 'R'.
    
# T: O(N)
# S: O(1)
    def sortSentence(self, s: str) -> str:
        s = s.split()
        output = ""
        obj = {}
        for i in s:
            rank = i[-1]
            p = i[:-1]
            obj[rank] = p
        obj = sorted(obj.items())
        for j in obj:
            output +=j[1]
            output +=" "
        return output.strip()
    
# Input: s = "is2 sentence4 This1 a3"
# Output: "This is a sentence"
# Explanation: Sort the words in s to their original positions "This1 is2 a3 sentence4", then remove the numbers.

# T: O(N)
# S: O(N)
    def isUgly(self, n: int) -> bool:
        while n>=1:
            if n==1:return True
            if n%2==0:n/=2
            elif n%3==0:n/=3
            elif n%5==0:n/=5
            else:return False
            
# T : O(N)
# S: O(1)
    def wordPattern(self, pattern: str, s: str) -> bool:
        mapping1 = {}
        mapping2 = {}
        words = s.split()
        if len(pattern) != len(words):
            return False
        
        i = 0
        while i < len(words):
            is_familiar1 = words[i] in mapping1.keys()
            is_familiar2 = pattern[i] in mapping2.keys()
            state = [is_familiar1, is_familiar2]
            match state:
                case [True, False] | [False, True]:
                    return False            
                
                case [False, False]:
                    mapping1[words[i]] = pattern[i]
                    mapping2[pattern[i]] = words[i]

                case [True, True]:
                    if mapping1[words[i]] != pattern[i]:
                        return False
            i += 1        
        return True          
    def canWinNim(self, n: int) -> bool:
        return n % 4
    def firstUniqChar(self, s: str) -> int:
        
        
        c = collections.Counter(list(s))
        
        for i in range(len(s)):
            
            if c.get(s[i]) == 1:
                return i
        
        return -1   
   # T: O(N)
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        coded = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",\
                 ".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",\
                 ".--","-..-","-.--","--.."]
        res = set()
        for i in words:
            code = ''
            for j in i :
                code += coded[ord(j)-97]
            res.add(code)
            
        return len(res) 
# T: O(N)
# S: O(1)
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        ransomNote=list(ransomNote)
        magazine=list(magazine)
        for i in ransomNote:
            if i in magazine:
                magazine.remove(i)
            else:return False
        return True  
    def countAndSay(self, n: int) -> str:
        res = '1'
        count = 0
        for i in range(1, n):
            temp = ''
            l = 0
            count = 0
            for r in range(len(res) + 1):
                if (r < len(res) and res[l] != res[r]) or r == len(res):
                    count = r - l
                    temp += str(count) + res[r - 1]
                    l = r
            
            res = temp
            
        return res
