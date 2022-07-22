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
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        res = []
        going_right = True
        queue = collections.deque([root])
        
        while queue:
            list_values = []
            for _ in range(len(queue)):                                
                if going_right:
                    top = queue.popleft()
                    list_values.append(top.val)
                    if top.left:
                        queue.append(top.left)
                    
                    if top.right:
                        queue.append(top.right)
                else:
                    top = queue.pop()
                    list_values.append(top.val)
                    if top.right:
                        queue.appendleft(top.right)                    
                    if top.left:
                        queue.appendleft(top.left)

            res.append(list_values)
            going_right = not going_right
        return res

# Input: root = [3,9,20,null,null,15,7]
# Output: [[3],[20,9],[15,7]]
    
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
class SparseVector:
  def __init(self, nums: List[int]):      
          self.nums = []
          for i, num in enumerate(nums):
              self.nums.append((i,num))
  def dotproduct(self, vec: SparseVector):      
          target = 0
          i,j = 0,0
          for i < len(nums) and j < len(vec):
              i_idx, i_num = self.nums[i]
              j_idx, j_num = self.vec[j]
              
              if i_idx == j_idx:
                  target += (i_num * j_num)
                  i += 1
                  j += 1
              elif i_idx < j_idx:
                  i += 1
              else:
                  j += 1
          return target
    def minRemoveToMakeValid(self, s: str) -> str:
        l, r =0,0
        res= []
        for i in s:
            if i == "(":
                l +=1
                res.append(i)
            elif i == ")":
                if l > r:
                    r +=1
                    res.append(i)
            else:
                res.append(i)
            
        if l == r:
            return "".join(res)
        else:
            output = []
            for i in range(len(res) - 1, -1, -1):
                curr = res[i]
                if curr == "(":
                    if l <= r:
                        output.append(curr)
                    else:
                        l -= 1
                elif curr == ")":
                    output.append(curr)
                else:
                    output.append(curr)
            
            return "".join(reversed(output))

# Input: s = "lee(t(c)o)de)"
# Output: "lee(t(c)o)de"
# Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.        
        
        
# T: O(2 * N) --> O(N)        
# S: O(N)
    def validPalindrome(self, s: str) -> bool:        
        l, r = 0, len(s) - 1
        
        while l < r:
            if s[l] != s[r]:
                return self.is_palindrome(s, l+1, r) or self.is_palindrome(s, l, r -1)
            l +=1
            r -=1
        return True
    
    
    def is_palindrome(self, s, l, r):            
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True

#Input: s = "abca"
#Output: true
#Explanation: You could delete the character 'c'.
        
# T: O(N)
# S: O(1)
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = defaultdict(list)
        
        for s in strs:
            count = [0] * 26
            
            for c in s:
                count[ord(c) - ord("a")] += 1
                
            res[tuple(count)].append(s)
        return res.values()

#Input: strs = ["eat","tea","tan","ate","nat","bat"]
#Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
        
# T: O(N) * O(K) -> K = Longest string length
# S: O(NK)
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        if root is None:
            return None
        if root.val == p.val or root.val == q.val:
            return root

        left = self.lowestCommonAncestor( root.left, p, q )
        right = self.lowestCommonAncestor( root.right, p, q )
                
        if left and right:
            return root
        if left:
            return left
        else:
            return right
          
# Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
# Output: 5
# Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
#T: O(N)
#S: O(1)
    def lowestCommonAncestor(self, p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        seen = set()
        
        while p:
            seen.add(p)
            p = p.parent()
        while q:
            if q in seen:
              return q
            q = q.parent()
#T: O(N)
#S: O(N)
    def lowestCommonAncestor(self, p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        p_copy, q_copy = p,q
        
        while p_copy != q_copy:
          p_copy = p.parent() if p else q
          q_copy = q.parent() if q else p 
          
        return p_copy

#T: O(N)
#S: O(1)
