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
