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
