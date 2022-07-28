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
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        if not k:
            return [target.val]
        
        queue = collections.deque([root])
        graph = collections.defaultdict(list)
        
        
        while queue:
            node = queue.popleft()
            
            if node.left:
                graph[node].append(node.left)
                graph[node.left].append(node)
                
                queue.append(node.left)
                
            if node.right:
                graph[node].append(node.right)
                graph[node.right].append(node)
                
                queue.append(node.right)
                
        
        queue = collections.deque([(target, 0)])
        visited = set([target])
        res = []
        
        while queue:
            node, distance = queue.popleft()
            
            if distance == k:
                res.append(node.val)
            else:
                for edge in graph[node]:
                    if edge not in visited:
                        visited.add(edge)
                        queue.append((edge, distance + 1))
                        
        return res
#Input: root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, k = 2
#Output: [7,4,1]
#Explanation: The nodes that are a distance 2 from the target node (with value 5) have values 7, 4, and 1.
# T: O(N) + O(N) -> O(N)
# S: O(N) + O(N) -> O(N)

    def preorder(self, root: 'Node') -> List[int]:
        
        if not root:
            return []
        
        res = [root.val]
        
        self.preorder_helper(root.children, res)
        
        return res
    
    def preorder_helper(self, root, res):
        
        for i in root:
            res.append(i.val)
            self.preorder_helper(i.children, res)
        
#Input: root = [1,null,3,2,4,null,5,6]
#Output: [1,3,5,6,2,4]

# T: O(N)
# S: O(1)
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
    def removeDuplicates(self, s: str) -> str:
        stack = []
        
        for i in s:
            if not stack:
                stack.append(i)
            else:
                if i == stack[-1]:
                    stack.pop()
                else:
                    stack.append(i)
        
        return "".join(stack)
                
# Input: s = "azxxzy"
# Output: "ay"
#T : O(N)
#S: O(N)
    def removeDuplicates(self, s: str, k: int) -> str:
        stack = []
        
        for char in s:
            if not stack:
                stack.append([char, 1])
            
            else:                
                if(char != stack[-1][0]):
                    stack.append([char, 1])
                else:
                    if stack[-1][1] + 1 == k:
                        stack.pop()
                    else:
                        stack[-1][1] += 1
        
        stack = [char * count for char, count in stack]
        return "".join(stack)
                    
#Input: s = "deeedbbcccbdaa", k = 3
#Output: "aa"
#Explanation: 
#First delete "eee" and "ccc", get "ddbbbdaa"
#Then delete "bbb", get "dddaa"
#Finally delete "ddd", get "aa"
        
#T: O(N) + O(N) + O(N) --> O(N)
#S: O(N)
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) <= 1:
            return intervals
        
        
        
        intervals.sort()
        
        res = [intervals[0]]
        
        for start, end in intervals[1:]:
            if start <= res[-1][1]:
                res[-1][1] = max(end, res[-1][1])
            else:
                res.append([start, end])
        return res
    
# Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
# Output: [[1,6],[8,10],[15,18]]
# Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
#T: O(NlogN) + O(N) -> O(NlogN)
#S: O(N)
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        i=0
        ans=0
        sign = 1
        MAX_ANS = 2 ** 31 - 1
        MIN_ANS = -2**31 
        
        
        while(i < len(s) and s[i] == ' '):
            i += 1
            
        if(i < len(s) and s[i] == '-'):
            i += 1
            sign = -1
            
        elif( i<len(s) and s[i] == '+'):
            i += 1

       
        digits = set('0123456789')
        while(i<len(s) and s[i] in digits):
            ans = ans * 10 + int(s[i])
            i+=1
            
        ans = sign*ans    
        if( ans < 0):
            return max(ans, MIN_ANS)
        return min(ans, MAX_ANS)
#Input: s = "4193 with words"
#Output: 4193    
    
#T : O(N)
#S : O(1)
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        self.memo = {}
        
        self.directions = {(1,0), (0,1), (-1,0), (0,-1)}
        
        res = 1
        
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                res = max(res, self.dfs(matrix, row, col))
        
        return res
    def dfs(self, matrix, cur_row, cur_col):
        if (cur_row, cur_col) in self.memo:
            return self.memo[(cur_row, cur_col)]
        
        self.memo[(cur_row, cur_col)] = 1
        
        for row_inc,col_inc in self.directions:
            new_row = cur_row + row_inc
            new_col = cur_col + col_inc
            
            if (0 <= new_row < len(matrix)) and (0<= new_col < len(matrix[0])) and matrix[cur_row][cur_col] < matrix[new_row][new_col]:
                self.memo[(cur_row,cur_col)] = max(self.memo[(cur_row,cur_col)] , 1 + self.dfs(matrix, new_row, new_col))
                
                
        return self.memo[(cur_row, cur_col)]
    
# Input: matrix = [[9,9,4],[6,6,8],[2,1,1]]
#Output: 4
#Explanation: The longest increasing path is [1, 2, 6, 9].
# T: O(R * C)                
# S: O(R * C)  
    def findMinDifference(self, timePoints: List[str]) -> int:
        for i, time in enumerate(timePoints):
            hours, minutes = time.split(":")
            
            min_past_midnight = int(hours) * 60 + int(minutes)
            
            timePoints[i] = min_past_midnight
            
        timePoints.sort()
        
        res = 1440 + timePoints[0] - timePoints[-1]
        
        for i in range(1, len(timePoints)):
            res = min(res, timePoints[i] - timePoints[i-1])
        
        return res
    
# Input: timePoints = ["23:59","00:00"]
# Output: 1
    
# T: O(N) + O(NlogN) + O(N)
# S: O(1) if overriding inputs otherwise O(N) 
    def minAddToMakeValid(self, s: str) -> int:
        l_count = r_count = added = 0
        
        for char in s:
            if char == "(":
                l_count +=1 
            else:
                if l_count > r_count:
                    r_count += 1
                else:
                    added += 1
                
        added += l_count - r_count
        
        return added
                    
#T: O(N)
#S: O(1)
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        queue = collections.deque([s])
        
        visited = set()
        
        while queue:
            word = queue.popleft()
            
            if word in visited :
                continue
            else:
                if not word:
                    return True
                visited.add(word)
                
                for start_word in wordDict:
                    if word.startswith(start_word):
                        queue.append(word[len(start_word):])
        return False
                        
# Input: s = "leetcode", wordDict = ["leet","code"]
# Output: true
# Explanation: Return true because "leetcode" can be segmented as "leet code".        
# T: O(N) * O(N) * O(N) -> O(N^3)
# S: O(N)
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = []
        
        for i,t in enumerate(temperatures):
            while stack and t > stack[-1][0]:
                stackT, stackI = stack.pop()
                res[stackI] = ( i - stackI)
            stack.append([t, i])
            
        return res
#Input: temperatures = [73,74,75,71,69,72,76,73]
#Output: [1,1,4,2,1,1,0,0]

#T: O(N)
#S: O(N)
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        if len(points) <= k:
            return points
        
        
        points_euc_dist = []
        
        for x,y in points:
            euc_dist = self.euc(x,y)
            points_euc_dist.append((euc_dist, (x,y)))
        
        heap = []
        
        for euc_dist, point in points_euc_dist:
            if len(heap) < k:
                heapq.heappush(heap, (-euc_dist,point))
            else:
                if -euc_dist > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (-euc_dist, point))
                    
        
        return [point for _dist, point in heap]
        
    def euc(self, x, y):
        return (math.sqrt(x**2 + y**2))
    
#Input: points = [[3,3],[5,-1],[-2,4]], k = 2
#Output: [[3,3],[-2,4]]
#Explanation: The answer [[-2,4],[3,3]] would also be accepted.    

#T: O(N*logK)
#S: O(N)
    def exist(self, board: List[List[str]], word: str) -> bool:
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == word[0]:
                    if self.dfs(board, row, col, word):
                        return True
        return False
    
    def dfs(self, board, row, col, word):
        if not word:
            return True
        
        if (0 <= row < len(board)) and (0 <= col < len(board[0])) and board[row][col] != "#" and board[row][col] == word[0]:
            placeholder = board[row][col]
            board[row][col] = "#"
            
            for inc_row, inc_col in [(0,1), (0,-1), (1,0), (-1,0)]:
                if self.dfs(board, row + inc_row, col + inc_col, word[1:]):
                    return True
            
            board[row][col] = placeholder
            return False
# Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
# Output: true 
# N: Rows * Col
# L: Length of Word
# T: O(N*3L)
# S: O(N)
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        song_dict = collections.defaultdict(int)
        
        pairs = 0
        
        for song_length in time:
            if not (song_length % 60):
                pairs += song_dict[0]
            
            else:
                pairs += song_dict[(60 - (song_length % 60))]
                
            song_dict[song_length % 60] += 1
        return pairs

# Input: time = [30,20,150,100,40]
# Output: 3
# Explanation: Three pairs have a total duration divisible by 60:
# (time[0] = 30, time[2] = 150): total duration 180
# (time[1] = 20, time[3] = 100): total duration 120
# (time[1] = 20, time[4] = 40): total duration 60    

# T: O(N)
# S: O(N)
    def longestValidParentheses(self, s: str) -> int:
        l_count = r_count = max_len = 0
        i = 0
        while i< len(s):
            if s[i] == "(":
                l_count += 1
            else:
                r_count += 1
                
            if l_count == r_count:
                max_len = max(max_len, l_count + r_count)
            elif r_count > l_count:
                l_count = r_count = 0
            
            i += 1
        l_count = r_count = 0 
        i = len(s) - 1
        
        while i >= 0:
            if s[i] == "(":
                l_count += 1
            else:
                r_count += 1
                
            if l_count == r_count:
                max_len = max(max_len, l_count + r_count)
            elif l_count > r_count:
                l_count = r_count = 0            
            
            i -= 1
            
        return max_len

# Input: s = "(()"
# Output: 2
# Explanation: The longest valid parentheses substring is "()".      
# T: O(N) + O(N) -> O(N)
# S: O(1)
    def sumZero(self, n: int) -> List[int]:
        res = []
        
        if not n:
            return res
        
        if n % 2:
            res.append(0)
            
        
        for i in range(1, n//2 + 1):
            res.append(i)
            res.append(-i)
            
        return res

# Input: n = 5
# Output: [-7,-1,1,3,4]
# Explanation: These arrays also are accepted [-5,-1,1,2,3] , [-3,-1,2,-2,4].    
    
#T: O(N//2) -> O(N)
#S: O(N)
    def reorganizeString(self, s: str) -> str:
        count = {}
        
        for sub in s:
            count[sub] = 1 + count.get(sub, 0)
        
        
        maxHeap = [[-cnt, char] for char, cnt in count.items()]
        heapq.heapify(maxHeap) # O(n)
        
        prev = None
        res = ""
        
        while maxHeap or prev:
            if prev and not maxHeap:
                return ""
            
            cnt, char = heapq.heappop(maxHeap)
            res += char
            cnt +=1 # its a min heap so we add, since the vals are negative, actually we want to subtract
            
            if prev:
                heapq.heappush(maxHeap, prev)
                prev = None
            if cnt !=0:
                prev = [cnt, char]
                
        return res
# Input: s = "aab"
# Output: "aba"

# T: O(NlogN)
# S: O(N)
    def findDuplicates(self, nums: List[int]) -> List[int]:
        res = []
        
        for num in nums:
            at_index = nums[abs(num) - 1]
            
            if at_index < 0:
                res.append(abs(num))
                
            else:
                nums[abs(num) - 1] *= -1
        return res
    
# Input: nums = [4,3,2,7,8,2,3,1]
# Output: [2,3]    
# T: O(N)
# S: O(1)           
def count_substring(string,sub_string):
    len1 = len(string)
    len2 = len(sub_string)
    j =0
    counter = 0
    while(j < len1):
        if(string[j] == sub_string[0]):
            if(string[j:j+len2] == sub_string):
                counter += 1
        j += 1

    return counter
# Input string = "ABCDCDC" substring = "CDC"
# Output 2

# T: O(N)
# S: O(1)
    def findMaxLength(self, nums: List[int]) -> int:
        seen_at = {}
        seen_at[0] = -1
        
        
        count = ans = 0 
        
        for i, num in enumerate(nums):
            count += 1 if num else -1
            
            if count in seen_at:
                ans = max(ans, i - seen_at[count])
                
            else:
                seen_at[count] = i
                
        return ans
    
# Input: nums = [0,1]
# Output: 2
# Explanation: [0, 1] is the longest contiguous subarray with an equal number of 0 and 1.    

# T: O(N)        
# S: O(N)
    def myPow(self, x: float, n: int) -> float:
        def helper(x,n):
            if x == 0: return 1
            if n == 1: return x
            res = helper(x , n // 2)
            res = res * res
            return x * res if n % 2 else res
        
        res = helper(x, abs(n))
        return res if n >= 0 else 1/res

#  Input: x = 2.00000, n = 10
# Output: 1024.00000  
# T : O(logN)    
# S : O(1)
    def getMaxLen(self, nums: List[int]) -> int:
        ans, pos, neg = 0, 0, 0
        
        for num in nums:
            if num > 0:
                pos = 1 + pos
                neg = 1 + neg if neg else 0
                
            if num < 0:
                pos, neg = 1 + neg if neg else 0, 1 + pos
                
            if num == 0:
                pos, neg = 0,0
            
            ans = max(ans, pos)
            
        return ans
        
# Input: nums = [1,-2,-3,4]
# Output: 4
# Explanation: The array nums already has a positive product of 24.    

# T: O(N)
# S: O(1)
    def minAreaRect(self, points: List[List[int]]) -> int:
        min_size = float('inf')
        visited = set()
        
        
        for x1, y1 in points:
            for x2, y2 in visited:
                if (x1, y2) in visited and (x2, y1) in visited:
                    size = abs(x2 - x1) * abs(y2 - y1)
                    
                    min_size = min(size, min_size)
                    
            visited.add((x1, y1))
                
        return min_size if min_size != float('inf') else 0
    
# Input: points = [[1,1],[1,3],[3,1],[3,3],[2,2]]
# Output: 4
# T: O(N^2)
# S: O(N)
    def reverseWords(self, s: str) -> str:
        return ' '.join(s.strip().split()[::-1])
        
# T: O(N)
# S: O(1)
    def maxProfit(self, prices: List[int]) -> int:
        minprof = float("inf")
        maxprof = 0
        
        for i in prices:
            if i < minprof:
                minprof = i
            elif i - minprof > maxprof:
                maxprof = i - minprof
            
        return maxprof
            
# T: O(N)
# S: O(1)
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None
        pointer = head
        while pointer != fast:
            pointer = pointer.next
            fast = fast.next
        return pointer

# T: O(N)    
# S: O(1)
    def longestPalindrome(self, s: str) -> int:
        hMap = {}
        count = 0
        
        for i in s:
            if i in hMap:
                hMap[i]  += 1
                if hMap[i] == 2:
                    count += 2
                    hMap[i] = 0
            else:
                hMap[i] = 1
            
        if 1 in hMap.values():
            return count + 1
        else:
            return count
# Input: s = "abccccdd"
# Output: 7
# Explanation: One longest palindrome that can be built is "dccaccd", whose length is 7.        

# T: O(N)
# S: O(1)
