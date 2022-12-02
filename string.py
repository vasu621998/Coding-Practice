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
    def percentageLetter(self, s: str, letter: str) -> int:
        count = 0
        for ele in s:
            if ele==letter:
                count+=1
        return count*100//len(s)        

# T: O(N*K)
# S: O(1)

#Input: s = "foobar", letter = "o"
#Output: 33
#Explanation:
#The percentage of characters in s that equal the letter 'o' is 2 / 6 * 100% = 33% when rounded down, so we return 33.
    def minWindow(self, s: str, t: str) -> str:
        counter_t, window = Counter(t), defaultdict(int)
        start = balance = 0
        res = [len(s), 2 * len(s)]

        for i, ch in enumerate(s):
            if ch in counter_t:
                window[ch] += 1
                if window[ch] <= counter_t[ch]:
                    balance += 1
                if balance < len(t):
                    continue
                while s[start] not in counter_t or window[s[start]] > counter_t[s[start]]:
                    if s[start] in counter_t:
                        window[s[start]] -= 1
                    start += 1
                if i + 1 - start <= res[1] - res[0]:
                    res = [start, i + 1]
        
        return s[res[0]:res[1]]    
    
# T : O(S + T)
# S : O(T)

# Input: s = "ADOBECODEBANC", t = "ABC"
# Output: "BANC"
# Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
    def restoreIpAddresses(self, s: str) -> List[str]:
        result = []
        
        def valid(octet):
            first = 0 < int(octet) <= 255 and octet[0] != "0"
            second = 0 == int(octet) and len(octet) == 1
            return first or second
        
        def backtrack(start = 0, segments = 4, current = []):
            if segments == 0:
                if start == len(s):
                    result.append(".".join(current))
                return

            for i in range(start+1, min(len(s) + 1, start + 4)):
                octet = s[start:i]
                current.append(octet)
                if valid(octet):
                    backtrack(i, segments - 1, current)
                current.pop()
        
        backtrack()
        
        return result
    
# T : O(1)
# S : O(1)

# Input: s = "25525511135"
# Output: ["255.255.11.135","255.255.111.35"]
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        s1_len, s2_len, s3_len = len(s1), len(s2), len(s3)
        if s1_len + s2_len != s3_len:
            return False

        dp = [[False] * (s2_len + 1) for _ in range(s1_len + 1)]
        dp[s1_len][s2_len] = True
        for i in range(s1_len, -1, -1):
            for j in range(s2_len, -1, -1):
                if i < s1_len and s1[i] == s3[i + j] and dp[i + 1][j]:
                    dp[i][j] = True
                if j < s2_len and s2[j] == s3[i + j] and dp[i][j + 1]:
                    dp[i][j] = True

        return dp[0][0]
    
# T : O(mn)
# S : O(mn)


# Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
# Output: true
# Explanation: One way to obtain s3 is:
# Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
# Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
# Since s3 can be obtained by interleaving s1 and s2, we return true.
    def breakPalindrome(self, palindrome: str) -> str:
        n = len(palindrome)
        if n == 1:
            return ''
        for i in range(n//2):
            if palindrome[i] != 'a':
                return palindrome[:i] + 'a' + palindrome[i+1:]
        return palindrome[:-1] + 'b'    
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        res = []
        if len(s) < 10:
            return res
        
        c = {}
        for i in range(len(s) - 9):
            dna = s[i:i+10]
            c[dna] = c.get(dna, 0) + 1
        
        for key in c:
            if c[key] > 1:
                res.append(key)
        
        return res
    
# T : O(N)
# S : O(N)

#Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
#Output: ["AAAAACCCCC","CCCCCAAAAA"]
    def checkIfPangram(self, sentence: str) -> bool:
        letters_dict = {}
	    # we keep track of what letters have appeared with letters_dict
        for letter in sentence:
            if letter not in letters_dict:
                letters_dict[letter] = 1
    
        # there are 26 english characters, if  letters_dict has at least 26 keys, then its good
        return len(letters_dict) >= 26        
    
# T : O(N)
# S : O(1)

#A pangram is a sentence where every letter of the English alphabet appears at least once.

#Given a string sentence containing only lowercase English letters, return true if sentence is a pangram, or false otherwise.
    def calculate(self, s: str) -> int:
        stack = [] 
        number, sign, res = 0, 1, 0

        for char in s: 
            if char.isdigit(): #
                number = number * 10 + int(char)
            elif char == '+' or char == '-':
                res += sign * number
                sign = 1 if char == '+' else -1
                number = 0
            elif char == '(':
                stack.append(res)
                stack.append(sign)
                res, sign = 0, 1
            elif char == ')':
                res += sign * number
                number = 0
                res *= stack.pop()
                res += stack.pop() 
            else:
                continue

        return res + (number * sign)
    
    
# T : O(N)
# S : O(N)
#Input: s = "1 + 1"
#Output: 2
	def calculate(self, s: str) -> int:
		s_len = len(s)
		stack_nums = list()
		stack_ops = list()
		curr_num = 0

		for i in range(s_len):
			if s[i] == ' ':
				continue 
			elif s[i] == '+' or s[i] == '-' or s[i] == '*' or s[i] == '/':
				stack_nums.append(curr_num)
				stack_ops.append(s[i])
				curr_num = 0
			else: 
				curr_num = curr_num*10 + int(s[i]) 

		stack_nums.append(curr_num)
		# print(stack_nums)
		# print(stack_ops)

		i = 0 
		while i<len(stack_ops):
			if stack_ops[i] == '*': 
				first = stack_nums.pop(i) 
				second = stack_nums.pop(i) 
				res = first*second
				stack_nums.insert(i, res)
				stack_ops.pop(i)
			elif stack_ops[i] == '/': 
				first = stack_nums.pop(i) 
				second = stack_nums.pop(i)
				res = first//second
				stack_nums.insert(i, res) 
				stack_ops.pop(i)
			else: 
				i += 1 
				continue
		# print(stack_nums)
		# print(stack_ops)
		res = stack_nums[0] 
		num_items = len(stack_nums)
		num_ops = len(stack_ops)
		for i in range(num_ops):
			if stack_ops[i] == '+': 
				res += stack_nums[i+1]
			elif stack_ops[i] == '-':
				res -= stack_nums[i+1]
		return res 
    
# T : O(N)
# S : O(N)

#Input: s = "3+2*2"
#Output: 7
    def getHint(self, secret: str, guess: str) -> str:
        # Dictionary for Lookup
        lookup = Counter(secret)
        
        x, y = 0, 0
        
        # First finding numbers which are at correct position and updating x
        for i in range(len(guess)):
            if secret[i] == guess[i]:
                x+=1
                lookup[secret[i]]-=1
        
        # Finding numbers which are present in secret but not at correct position 
        for i in range(len(guess)):
            if guess[i] in lookup and secret[i] != guess[i] and lookup[guess[i]]>0:
                y+=1
                lookup[guess[i]]-=1
        
		# The reason for using two for loop is in this problem we have 
		# to give first priority to number which are at correct position,
		# Therefore we are first updating x value
		
        return "{}A{}B".format(x, y)        
    
# T : O(N)
# S : O(1)


# Input: secret = "1807", guess = "7810"
# Output: "1A3B"
# Explanation: Bulls are connected with a '|' and cows are underlined:
# "1807"
  |
# "7810"
    def makeGood(self, s: str) -> str:
        stack=[]
        for x in s:
            stack.append(x)
            if len(stack)>=2 and stack[-1].lower()==stack[-2].lower() and stack[-1]!=stack[-2]:
                stack.pop()
                stack.pop()
        return "".join(stack)  
    
# T : O(N)
# S : O(N)


#Input: s = "leEeetcode"
#Output: "leetcode"
#Explanation: In the first step, either you choose i = 1 or i = 2, both will result "leEeetcode" to be reduced to "leetcode".
    def removeDuplicateLetters(self, s: str) -> str:
        d=[0 for _ in range(26)]
        for item in s:
            j=ord(item)-97
            d[j]+=1
        stack=[]
        d1=[False for _ in range(26)]
        for item in s:
            if(len(stack)==0 or stack[-1]<=item):
                if(not d1[ord(item)-97]):
                    stack.append(item)
                d[ord(item)-97]-=1
                d1[ord(item)-97]=True
            else:
                if(not d1[ord(item)-97]):     
                    while(len(stack)!=0 and stack[-1]>item):
                        j=ord(stack[-1])-97
                        if(d[j]==0):
                            break
                        d1[ord(stack[-1])-97]=False
                        stack.pop()
                    stack.append(item)
                d[ord(item)-97]-=1
                d1[ord(item)-97]=True
        return "".join(stack)
    
# T : O(N)
# S : O(N)

#Input: s = "bcabc"
#Output: "abc"
    def closeStrings(self, word1: str, word2: str) -> bool:
        c1=Counter(word1)
        c2=Counter(word2)
        if c1.keys()!=c2.keys():
            return False
        v1=list(c1.values())
        v2=list(c2.values())
        v1.sort()
        v2.sort()
        return v1==v2
    
# T : O(NlogN)
# S  : O(1)

#Input: word1 = "abc", word2 = "bca"
#Output: true
#Explanation: You can attain word2 from word1 in 2 operations.
#Apply Operation 1: "abc" -> "acb"
#Apply Operation 1: "acb" -> "bca"
