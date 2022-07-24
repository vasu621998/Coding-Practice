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

    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        column = collections.defaultdict(list)
        queue = collections.deque([(0,root)])
        
        res = []
        
        min_x = float('inf')
        max_x = float('-inf')
        

        
        while queue:
            
            
            x, node = queue.popleft()
            
            column[x].append(node.val)
            
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            
            if node.left:
                queue.append((x-1, node.left))
        
            if node.right:
                queue.append((x+1, node.right))        
                            
        
        for i in range(min_x, max_x +1):
            res.append(column[i])
        return res

    
# Input: root = [1,2,3,4,5,6,7]
#Output: [[4],[2],[1,5,6],[3],[7]]    
# T: O(N)            
# S: O(N)
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if not root:
            return 0
        ans = 0
        
        stack = [root]
        
        while stack:
            node = stack.pop()
            if node:  
                if low <= node.val <= high:
                    ans += node.val
            
                if node.val > low:
                    stack.append(node.left)

                if node.val < high:
                    stack.append(node.right)
        return ans
# Input: root = [10,5,15,3,7,13,18,1,null,6], low = 6, high = 10
#Output: 23    
# T: O(N)
# S: O(N)
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if not root:
            return 0
        ans = 0
        self.range_sum = 0
        self.dfs(root, low,high)
        return self.range_sum
    def dfs(self, node, low, high):
        if node:
            if low <= node.val <= high:
                self.range_sum += node.val
            if node.val > low:
                self.dfs(node.left, low, high)
            if node.val < high:
                self.dfs(node.right, low, high)           
# Input: root = [10,5,15,3,7,13,18,1,null,6], low = 6, high = 10
#Output: 23    
# T: O(N)
# S: O(N)
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        self.max_path = root.val
        
        self.dfs(root)
        
        return self.max_path
    
    def dfs(self, node):
        if not node:
            return 0
        
        #Checking for leaf node
        if not node.left and not node.right:
            self.max_path = max(node.val, self.max_path)
            return node.val
    
        left = self.dfs(node.left)
        right = self.dfs(node.right)
        
        self.max_path = max(self.max_path, 
                            node.val + left + right,
                            node.val + left,
                            node.val + right,
                            node.val)
        
        return max(node.val, 
                   node.val + left, 
                   node.val + right, 
                   0)
    
# Input: root = [-10,9,20,null,null,15,7]
# Output: 42
# Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
# T: O(N)
# S: O(1) If not counting recursive stack space, else O(N) if counting recursive stack space

    def goodNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        self.good_nodes = 0
        self.dfs(root,root.val)
        
        return self.good_nodes
    
    
    def dfs(self, node, max_val):
        if not node:
            return 
        
        if node.val >= max_val:
            self.good_nodes += 1
            
            max_val = node.val
            
        self.dfs(node.left, max_val)
        self.dfs(node.right, max_val)
            
#Input: root = [3,1,4,3,null,1,5]
#Output: 4
#Explanation: Nodes in blue are good.
#Root Node (3) is always a good node.
#Node 4 -> (3,4) is the maximum value in the path starting from the root.
#Node 5 -> (3,4,5) is the maximum value in the path
#Node 3 -> (3,1,3) is the maximum value in the path.

#T: O(N)
#S: O(1) if not counting recursive stack frames otherwise O(N)
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
    def lowestCommonAncestor(self, p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        self.p_visited, self.q_visited = False, False
        ans = self.dfs(root)
        return ans if p_visited and q_visited else None
        
    def dfs(self, node, p,q):
        if not node:
            return None
        l = self.dfs(node.left)
        r = self.dfs(node.right)
        
        if node == p or node == q:
            if node == p:
                self.p_visited = True
            else:
                self.q_visited = True
            
            return node
          
        if l and r:
            return node
        return l or r
      
#T: O(N)
#S: O(N) if couting stack frames else O(1)
    def lowestCommonAncestor(self, root: 'TreeNode', nodes: 'TreeNode') -> 'TreeNode':
        nodes = set(nodes)
        
        def dfs(node):
            if not node:
                return None
            if node in nodes:
                return node
            l = dfs(node.left)
            r = dfs(node.right)
            
            if l and r:
                return node
             else:
                return l or r
         return dfs(root)
      
      
#T: O(N) + O(N) -> O(N)
#S: O(N) + O(N) --> O(N) if couting stack frames else O(1)

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        curr = dummy
        carry = 0
        
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            
            val = v1 + v2 + carry
            
            carry = val // 10
            val = val % 10
            
            curr.next = ListNode(val)
            
            curr = curr.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            
        return dummy.next
# Input: l1 = [2,4,3], l2 = [5,6,4]
# Output: [7,0,8]
# Explanation: 342 + 465 = 807.        
# T: O(N+M)
# S: O(N+M)
    def letterCombinations(self, digits: str) -> List[str]:
        res = []
        digitToChar = {
            "2" : "abc",
            "3" : "def",
            "4" : "ghi",
            "5" : "jkl",
            "6" : "mno",
            "7" : "pqrs",
            "8" : "tuv",
            "9" : "wxyz"
        }
        def backTracking(i, currentStr):
            if len(currentStr) == len(digits):
                res.append(currentStr)
                return
            for c in digitToChar[digits[i]]:
                backTracking(i+1, currentStr+c)
        if digits:
            backTracking(0, "")
        
        return res

    
# Input: digits = "23"
# Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]    
# T: O(4^N * N)
# S: O(N)
    def customSortString(self, order: str, s: str) -> str:
        s_count = collections.Counter(s)
        res = []
        for char in order:
            if char in s_count:
                res.extend(char * s_count[char])
                
                del s_count[char]
                
        for i in s_count:
            res.extend(i * s_count[i])
        
        return "".join(res)

# Input: order = "cba", s = "abcd"
# Output: "cbad"    
# T: O(M + N) -> O(N)
# S: O(M + N) -> O(N)
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        l = max(weights)
        r = sum(weights)
        
        while l < r:
            mid = (l + r) // 2
            
            if self.can_ship(mid, weights, days):
                r = mid
            else:
                l = mid + 1
            
        return r
        
    def can_ship(self, candidates, weights, days):
        days_taken = 1
        curr_weight = 0
        
        for w in weights:
            curr_weight += w
            if curr_weight > candidates:
                days_taken += 1
                curr_weight = w
        
        return days_taken <= days
    
#Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
#Output: 15
#Explanation: A ship capacity of 15 is the minimum to ship all the packages in 5 days #like this:
#1st day: 1, 2, 3, 4, 5
#2nd day: 6, 7
#3rd day: 8
#4th day: 9
#5th day: 10 

# T: O(NLOGN)
# S: O(1)

    def minDeletions(self, s: str) -> int:
        
        
        deletion = 0
        myset = collections.Counter(s)
        freqset = set()
        
        for i, count in myset.items():
            while count > 0 and count in freqset:
                count -= 1
                deletion += 1
            
            freqset.add(count)
            
        return deletion
# Input: s = "aaabbbcc"
# Output: 2
#Explanation: You can delete two 'b's resulting in the good string "aaabcc".
#Another way it to delete one 'b' and one 'c' resulting in the good string "aaabbc".    
# T: O(N) + O(N) -> O(N)
# S: O(1) + O(N) -> O(N)
    def simplifyPath(self, path: str) -> str:
        path_items =  path.split("/")
        stack = []
        
        for item in path_items:
            if item == "." or not item:
                continue
            elif item == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(item)
        return "/" + "/".join(stack)
                
#Input: path = "/home/"
#Output: "/home"
#Explanation: Note that there is no trailing slash after the last directory name.
        
# T: O(N) + O(N) + O(N) -> O(N)
# S: O(N) (Need to maintain a stack)
