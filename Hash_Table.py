class MyHashSet:

    def __init__(self):
        self.array = [[] for _ in range(1000)]
        

    def add(self, key: int) -> None:
        subkey = key % 1000
        if not self.contains(key):
            self.array[subkey].append(key)

    def remove(self, key: int) -> None:
        subkey = key % 1000
        if self.contains(key):
            self.array[subkey].remove(key)        

    def contains(self, key: int) -> bool:
        subkey = key % 1000
        return key in self.array[subkey]


# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)
class MyHashMap:

    def __init__(self):
        self.l = [-1 for _ in range(1000001)]

    def put(self, key: int, value: int) -> None:
        self.l[key] = value

    def get(self, key: int) -> int:
        return self.l[key]

    def remove(self, key: int) -> None:
        self.l[key] = -1


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for n in nums:
            res = n ^ res
        return res
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        res = []
        
        for x in nums1:
            for y in nums2:
                if x == y:
                    res.append(x)
        return set(res)
    def isHappy(self, n: int) -> bool:
        visit = set()
        
        while n not in visit:
            visit.add(n)
            n = self.sumofSquare(n)
            
            if n == 1:
                return True
        return False
        
    def sumofSquare(self, n: int) -> int:
        output = 0
        while n:
            digit = n % 10
            digit = digit ** 2
            output += digit
            n = n // 10
        return output
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        mymap = {}
        
        for i, n in enumerate(nums):
            diff = target - n
            if diff in mymap:
                return [mymap[diff], i]
            mymap[n] = i
    def isIsomorphic(self, s: str, t: str) -> bool:
        ST, TS = {}, {}
        
        for i in range(len(s)):
            c1, c2 = s[i], t[i]
            
            if((c1 in ST and ST[c1] != c2) or (c2 in TS and TS[c2] != c1)):
                return False
            
            ST[c1] = c2
            TS[c2] = c1
        return True
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        hashmap = {}
        for i in range(len(list1)):   #step 1
            hashmap[list1[i]] = i
        
        res = []

        minsum = float("inf")   #step 2
        
        for j in range(len(list2)):    #step 3
            if list2[j] in hashmap:
                Sum = j + hashmap[list2[j]]    #step 3a
                
                if Sum < minsum:   #step 3b
                    minsum = Sum
                    res = []
                    res.append(list2[j])
                elif Sum == minsum:     #step 3c
                    res.append(list2[j])
        
        return res
    def firstUniqChar(self, s: str) -> int:
        import collections
        
        c = collections.Counter(list(s))
        
        for i in range(len(s)):
            
            if c.get(s[i]) == 1:
                return i
        
        return -1
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        arr = []
        i, j = 0, 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                i += 1
            elif nums2[j] < nums1[i]:
                j += 1
            else:
                arr.append(nums1[i])
                i += 1
                j += 1
        return arr
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        prevMap = {}#val|index
        
        
        for i,n in enumerate(nums):
            if nums[i] in prevMap:
                if k >= abs(i-prevMap[n]):
                    return True       
            prevMap[n] = i
        return False 
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = defaultdict(list)
        
        for s in strs:
            count = [0] * 26
            
            for c in s:
                count[ord(c) - ord("a")] += 1
                
            res[tuple(count)].append(s)
        return res.values()
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        cols = collections.defaultdict(set)
        rows = collections.defaultdict(set)
        squares = collections.defaultdict(set)
        
        
        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                if (board[r][c] in rows[r] or 
                    board[r][c] in cols[c] or 
                    board[r][c] in squares[(r//3, c//3)]):
                    return False
                cols[c].add(board[r][c])
                rows[r].add(board[r][c])
                squares[(r//3, c//3)].add(board[r][c])
        return True
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        jewels=[ord(x) for x in jewels]
        res=0
        for x in stones:
            if(ord(x) in jewels):
                res+=1
        return res
    def lengthOfLongestSubstring(self, s: str) -> int:
        cset = set()
        l = 0
        res = 0
        for r in range(len(s)):
            while s[r] in cset:              
                cset.remove(s[l])
                l += 1
            cset.add(s[r])
            res  = max(res, r-l+1)
        return res
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        n=len(nums1)
        res=0
        d1=defaultdict(int)
        d2=defaultdict(int)
        
        for i in range(n):
            for j in range(n):
                d1[nums1[i]+nums2[j]]+=1
        
        for i in range(n):
            for j in range(n):
                d2[nums3[i]+nums4[j]]+=1
        
        for key in d1:
            res+=(d1[key]*d2[-key])
        
        return res
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}
        freq = [[] for i in range(len(nums) + 1)]
        
        for n in nums:
            count[n] = 1 + count.get(n,0)
        
        for n,c in count.items():
            freq[c].append(n)
            
        
        res = []
        
        for i in range(len(freq)-1, 0, -1):
            for n in freq[i]:
                res.append(n)
                if len(res) ==k:
                    return res
        
class RandomizedSet:


    def __init__(self):
        self.val_num = dict()
        self.num_val = dict()
        self.length = 0

    def insert(self, val: int) -> bool:
        if val in self.val_num:
            return False
        self.length += 1
        self.val_num[val] = self.length
        self.num_val[self.length] = val
        return True

    def remove(self, val: int) -> bool:
        if val not in self.val_num:
            return False
        num = self.val_num[val]
        if num == self.length:
            del self.val_num[val]
            del self.num_val[num]
        else:
            last = self.num_val[self.length]
            self.val_num[last] = num
            del self.val_num[val]
            self.num_val[num] = last
            del self.num_val[self.length]
        self.length -= 1
        return True

    def getRandom(self) -> int:
        return self.num_val[randint(1, self.length)]
