class Solution:
  def depthSum(self, nestedList: List[NestedInteger]) -> int:      
          res = 0
          depth = 1
          que = collections.deque(nestedList)
          while que:  
              cur = que.popleft()
              for elem in nestedList:                
                  if elem.isInteger():
                     res += elem.getInteger() * depth
                  else:
                     que.extend(cur.getList())
              depth += 1
          return res
    # BFS
    # T: O(N)
    # S: O(N)
    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = collections.defaultdict(list)
        
        def dfs(node, layer):
            if not node:
                return layer
            
            left = dfs(node.left, layer)
            right = dfs(node.right, layer)
            
            layer = max(left, right)
            
            res[layer].append(node.val)
            
            return layer + 1
    
        dfs(root, 0)
        
        return res.values()
    # DFS
    # T: O(N)
    # S: O(N)
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        queue = collections.deque([root])
        res = []
        
        while queue:
            length = len(queue)
            for i in range(length):
                node = queue.popleft()

                
                if node.left:
                    queue.append(node.left)
                
                if node.right:
                    queue.append(node.right)
                
                
                if i == length - 1:
                    res.append(node.val)
                    
        return res  
    # DFS
    # T: O(N)
    # S: O(N)
    def leftSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        queue = collections.deque([root])
        res = []
        
        while queue:
            length = len(queue)
            for i in range(length):
                node = queue.popleft()

                
                if node.left:
                    queue.append(node.left)
                
                if node.right:
                    queue.append(node.right)
                
                
                if i == 0:
                    res.append(node.val)
                    
        return res
    # DFS
    # T: O(N)
    # S: O(N)
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        self.res = []
        def dfs(node, level):
            if not node:
                return
            
            if len(self.res) <= level:
                self.res.append([node.val])
            else:
                self.res[level].append(node.val)
            dfs(node.left, level+1)
            dfs(node.right, level+1)
        dfs(root,0)
        return self.res
    # DFS
    # T: O(N)
    # S: O(N)
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1

        
        
        def alphanum(c):
            return (ord("A") <= ord(c) <= ord("Z") or 
             ord("a") <= ord(c) <= ord("z") or
             ord("0") <= ord(c) <= ord("9"))
        
        while l < r:
            while l<r and not alphanum(s[l]):
                l += 1
            while l<r and not alphanum(s[r]):
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l,r = l+1, r-1
        return True
# T: O(N)
# S: O(1)
    def isPalindrome(self, s: str) -> bool:
        mystr = ""
        for c in s:
            if c.isalnum():
                mystr += c.lower()
        return mystr == mystr[::-1]
            
# T: O(Nlogn)
# S: O(1)
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        
        for num in nums:
            if len(heap) <= k:
                heapq.heappush(heap, num)
            else:
                if heap[0] < num:
                    heapq.heappop(heap)
                    heapq.heappush(heap, num)
        return heap[0]
    
# T: O(NlogK)
# S: O(K)
