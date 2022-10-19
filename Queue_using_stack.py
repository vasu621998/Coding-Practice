class MyQueue:

    def __init__(self):
        self.s1 = []
        self.s2 = []
        

    def push(self, x: int) -> None:
        while self.s1:
            self.s2.append(self.s1.pop())
        
        self.s1.append(x)
        
        while self.s2:
            self.s1.append(self.s2.pop())

    def pop(self) -> int:
        return self.s1.pop()

    def peek(self) -> int:
        return self.s1[-1]
        

    def empty(self) -> bool:
        return not self.s1
   
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        countMap = {}
        for word in words:
            countMap[word] = countMap.get(word, 0) + 1
        
		# Use negative count to make it a max heap
        a = [(-1 * countMap[word], word) for word in countMap.keys()]
        
        heapq.heapify(a)
        
        res = [heapq.heappop(a)[1] for _ in range(k)]
        
		# Sort by count first, and lexicographically if count is same
        return sorted(res, key=lambda x: (-countMap[x], x))        
    
# T : O(NlogK)
# S : O(1)

#Input: words = ["i","love","leetcode","i","love","coding"], k = 2
#Output: ["i","love"]
#Explanation: "i" and "love" are the two most frequent words.
#Note that "i" comes before "love" due to a lower alphabetical order.
