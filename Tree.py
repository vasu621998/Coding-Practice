# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:mid+1], inorder[:mid])
        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])
        return root
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder or not postorder:
            return None
        
        root = TreeNode(postorder.pop())
        mid = inorder.index(root.val)
        
        root.right = self.buildTree(inorder[mid+1:], postorder)
        root.left = self.buildTree(inorder[:mid], postorder)
        return root
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        cur, nxt = root, root.left if root else None
        
        while cur and nxt:
            cur.left.next = cur.right
            if cur.next:
                cur.right.next = cur.next.left
                
            cur = cur.next
            
            if not cur:
                cur = nxt
                nxt = cur.left
                
        return root
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        curr = root
        
        while curr:
            if p.val > curr.val and q.val > curr.val:
                curr = curr.right
            elif p.val < curr.val and q.val < curr.val:
                curr = curr.left
            else:
                return curr
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        curr = root
        while curr:
            if p.val > curr.val and q.val > curr.val:
                curr = curr.right
            elif p.val < curr.val and q.val < curr.val:
                curr = curr.left
            else:
                return curr
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
    def serialize(self, root):
        res = []
        
        def dfs(node):
            if not node:
                res.append("N")
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ",".join(res)

    def deserialize(self, data):
        vals = data.split(",")
        self.i = 0
        
        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return None
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left = dfs()
            node.right = dfs()
            return node
        
        return dfs()
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        def helper(root, val):
            
            if not root:
                return None
            elif root.val == val:
                return root
            elif root.val < val:
                return helper(root.right, val)
            elif root.val > val:
                return helper(root.left, val)
            
        return helper(root, val)  
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def valid(node, left, right):
            if not node:
                return True
            if not (node.val < right and node.val > left):
                return False
            
            return valid(node.left, left, node.val) and valid(node.right, node.val, right)
        
        return valid(root, float("-inf"), float("inf"))
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        queue = [root]
        while queue:
            curr = 0
            for i in range(len(queue)):
                node = queue.pop(0)
                curr = curr + node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return curr

# BFS
#Input: root = [1,2,3,4,5,null,6,7,null,null,null,null,8]
#Output: 15

# T: O(N)
# S: O(N)
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        # DFS solution
        output = []
        stack = [(root, '')]
        
        while stack:
            node, path = stack.pop()
            path += str(node.val)
            
            if not node.left and not node.right:
                output.append(path)
                
            path += '->'
            if node.left:
                stack.append((node.left, path))
            if node.right:
                stack.append((node.right, path))
                
        return output    

#Input: root = [1,2,3,null,5]
#Output: ["1->2->5","1->3"]
    
# T: O(N)
# S: O(N)
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def traversal(left, right):
            if left > right: # If left greater than right, it means that added all subtree values
                return
            index = (left + right) // 2 # Getting the middle value for the subtree root value
            node = TreeNode(nums[index])
            node.left = traversal(left, index-1)
            node.right = traversal(index+1, right)
            return node
        return traversal(0, len(nums) - 1)   
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        res = [0]
        
        def dfs(root):
            if not root:
                return -1
            l = dfs(root.left)
            r = dfs(root.right)
            
            res[0] = max(res[0], l + r + 2)
            
            
            return 1 + max(l, r)
        
        dfs(root)
        return res[0]
    
# T: O(N)
# S: O(1)
    def checkTree(self, root: Optional[TreeNode]) -> bool:
        return root.val == root.left.val + root.right.val 
# T: O(N)
# S: O(1)
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        curr = root
        while curr:
            if p.val > curr.val and q.val > curr.val:
                curr = curr.right
            elif p.val < curr.val and q.val < curr.val:
                curr = curr.left
            else:
                return curr        
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        def DFS(node1,node2):
            if node1==target:
                return node2
            if node1 and node1.left is None and node1.right is None:
                return
            
            res1 = DFS(node1.left,node2.left) if node1 else None
            if res1 is not None:
                return res1
            res2 = DFS(node1.right,node2.right) if node1 else None
            if res2 is not None:
                return res2
        res=DFS(original,cloned)
        return res 
#T: O(N)
#S: O(1)
    def findTilt(self, root: Optional[TreeNode]) -> int:
        res = [0]
        def tilt_helper(root,res):
            if not root: 
                return 0
                
            left = tilt_helper(root.left,res)
            right = tilt_helper(root.right,res)

            res[0] += abs(left-right)

            return left + right + root.val

        tilt_helper(root,res)
        return res[0]
# T: O(N)
# S: O(1)
#Input: root = [1,2,3]
#Output: 1
#Explanation: 
#Tilt of node 2 : |0-0| = 0 (no children)
#Tilt of node 3 : |0-0| = 0 (no children)
#Tilt of node 1 : |2-3| = 1 (left subtree is just left child, so sum is 2; right #subtree is just right child, so sum is 3)
#Sum of every tilt : 0 + 0 + 1 = 1
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        res = defaultdict(int)
        cnt = defaultdict(int)
        def dfs(root, level):
            if not root: return
            res[level] += root.val
            cnt[level] += 1
            dfs(root.left, level + 1)
            dfs(root.right, level + 1)
        dfs(root, 0)
        return [res[i]/cnt[i] for i in res]
    
# T: O(N)
# S: O(1)

#Input: root = [3,9,20,null,null,15,7]
#Output: [3.00000,14.50000,11.00000]
#Explanation: The average value of nodes on level 0 is 3, on level 1 is 14.5, and on #level 2 is 11.
#Hence return [3, 14.5, 11].
	def averageOfSubtree(self, root: Optional[TreeNode]) -> int:
		tot_sum, tot_nodes, op = self.averageOfSubtree_helper(root)
		return op

	def averageOfSubtree_helper(self, root: Optional[TreeNode]) -> int:
		 # Base cases
		 
		if root == None:
			return 0, 0, 0
		
		# If the node is a leaf
		if root.left == None and root.right == None:
			return root.val,1,1
		
		# Recursively calling the left and right subtrees 
		lst_sum, lst_nodes, lst_op = self.averageOfSubtree_helper(root.left)
		rst_sum, rst_nodes , rst_op = self.averageOfSubtree_helper(root.right)
		
		# If the subtree sum == node's value, we add that node to the final output, else we wont 
		if (lst_sum + rst_sum + root.val)//(lst_nodes + rst_nodes + 1) == root.val:

			return lst_sum + rst_sum + root.val, lst_nodes + rst_nodes + 1, lst_op + rst_op + 1
		
		
		else:

			return lst_sum + rst_sum + root.val, lst_nodes + rst_nodes + 1, lst_op + rst_op
  
  
  # Time Complexity analysis :- 
  
  #  For every node, we check if it's value is equal to the subtree avg and return 3 values, which is a constant work, say k1 
  #  In that way, we check for the entire 'n' no.of nodes, so total work = n*k1, so
  
  # T: O(n)
  # S: O(1)
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        
        ans = []
        def helper(node):
            if not node:
                return 
            if node.val%2==0:
                if node.left :
                    if node.left.left:
                        ans.append(node.left.left.val)
                    if node.left.right:
                        ans.append(node.left.right.val)
                if node.right:
                    if node.right.left:
                        ans.append(node.right.left.val)
                    if node.right.right:
                        ans.append(node.right.right.val)
                        
            helper(node.left)
            helper(node.right)
        helper(root)
        return sum(ans)        

# T: O(N)
# S: O(1)

#Input: root = [6,7,8,2,7,1,3,9,null,1,4,null,null,null,5]
#Output: 18
#Explanation: The red nodes are the nodes with even-value grandparent while the blue nodes are the even-value grandparents.
        if not root: return None
        root.left, root.right = self.pruneTree(root.left), self.pruneTree(root.right)
        return root if (root.left or root.right or root.val == 1) else None
# T: O(N)

#Input: root = [1,null,0,0,1]
#Output: [1,null,0,null,1]
#Explanation: 
#Only the red nodes satisfy the property "every subtree not containing a 1".
#The diagram on the right represents the answer.
    def tree2str(self, root: Optional[TreeNode]) -> str:
        result=[]
        def createString(root):
            if not root:
                return
            result.append('(')
            result.append(str(root.val))
            createString(root.left)
            if not root.left and root.right:
                    result.append('()')
            createString(root.right)
            result.append(')')
        createString(root)
        return ''.join(result)[1:-1]   
# T: O(N)
# S: O(1)

#Input: root = [1,2,3,4]
#Output: "1(2(4))(3)"
#Explanation: Originally, it needs to be "1(2(4)())(3()())", but you need to omit all the unnecessary empty parenthesis pairs. And it will be "1(2(4))(3)"
    def bstToGst(self, root: TreeNode) -> TreeNode:
        def solve(root, currSum):
            if root:
                solve(root.right, currSum)
                currSum[0] += root.val
                root.val = currSum[0]
                solve(root.left, currSum)
        
        currSum = [0]
        solve(root, currSum)
        return root

# T: O(N)
# S: O(1)

# Input: root = [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
# Output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        def maxTree(nums):
            idx=nums.index(max(nums))
            node = TreeNode(nums[idx])
            if len(nums[idx+1:]) > 0:
                node.right = maxTree(nums[idx+1:])
            if len(nums[:idx]) > 0:
                node.left = maxTree(nums[:idx])
            return node
        return maxTree(nums)

# T: O(N)
# S: O(1)


#Input: nums = [3,2,1,6,0,5]
#Output: [6,3,5,null,2,0,null,null,1]
#Explanation: The recursive calls are as follow:
#- The largest value in [3,2,1,6,0,5] is 6. Left prefix is [3,2,1] and right suffix is #[0,5].
    #- The largest value in [3,2,1] is 3. Left prefix is [] and right suffix is [2,1].
    #    - Empty array, so no child.
    #    - The largest value in [2,1] is 2. Left prefix is [] and right suffix is [1].
    #        - Empty array, so no child.
    #        - Only one element, so child is a node with value 1.
    #- The largest value in [0,5] is 5. Left prefix is [0] and right suffix is [].
        #- Only one element, so child is a node with value 0.
        #- Empty array, so no child.
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        count = 0
        # root = [2,3,1,3,1,null,1]

        stack = [(root, 0)]
        # path = 00000000
        # nade = 2
        # 1 << node.val = 1 << 2 = 00000100
        # path = 00000100
        # node = 3
        # 1 << node.val = 00001000
        # path = 00001100
        # node =3
        # 1 << node.val = 00001000
        # path = 00000100
        # path & (path -1)
        # path -1 = 00000011
        # count =1

        while stack:
            node, path = stack.pop()
            if node is not None:
                path = path ^ (1 << node.val)
                # if it's a leaf, check if the path is pseudo-palindromic
                if node.left is None and node.right is None:
                    # check if at most one digit has an odd frequency
                    if path & (path - 1) == 0:
                        count += 1
                else:
                    stack.append((node.left, path))
                    stack.append((node.right, path))

        return count
    def numTrees(self, n: int) -> int:
        d = [0] * (n + 1)
        d[0], d[1] = 1, 1
        for i in range(2, n + 1):
            for j in range(i):
                d[i] += d[j] * d[i - j - 1]
        return d[-1]
    
    
# T : O(N)
# S : O(1)

# Input: n = 3
# Output: 5
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        inorder, stack = [], []

        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            
            node = stack.pop()
            inorder.append(node)
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
                    
        left = None
        for i in range(len(inorder) - 1):
            print(inorder[i].val, inorder[i + 1].val)
            if inorder[i].val > inorder[i + 1].val:
                left = inorder[i]
                break
                
        right = None
        for i in range(len(inorder) - 1, 0, -1):
            if inorder[i].val < inorder[i - 1].val:
                right = inorder[i]
                break

        left.val, right.val = right.val, left.val
        
# T : O(N)
# S : O(N)

# Input: root = [1,3,null,null,2]
# Output: [3,1,null,null,2]
# Explanation: 3 cannot be a left child of 1 because 3 > 1. Swapping 1 and 3 makes the BST valid.
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        
        # if root is None return empty list
        if root is None:
            return result
        
        # let's create a queue
        queue = []
        # add root node to it
        queue.append(root)
        
        # work until we have some in queue 
        while queue:
            
            # calculate the count of nodes on queue
            count = len(queue)
            
            # array to store nodes of a level
            l = []
            
            # for each node in a level
            while count > 0: 
                
                # pop and add to the level list
                temp = queue.pop(0)
                l.append(temp.val)
                
                # add if the node has left
                if temp.left:
                    queue.append(temp.left)
                # add if the node has right
                if temp.right:
                    queue.append(temp.right)

                count -= 1
            # add nodes of level into the result
            result.append(l)
        
        return list(reversed(result))
    
# T : O(N)
# S : O(1)

# Input: root = [3,9,20,null,null,15,7]
# Output: [[15,7],[9,20],[3]]
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        def BSTCreation(start,end):
            if start==end:
                return None
            slow = start
            fast = start.next
            while fast and fast.next and fast!=end and fast.next!=end:
                slow = slow.next
                fast = fast.next.next
            return TreeNode(slow.val,BSTCreation(start,slow),BSTCreation(slow.next,end))
        
        return BSTCreation(head,None)
    
# T : O(N)
# S : O(D)

# Input: head = [-10,-3,0,5,9]
# Output: [0,-3,9,-10,null,5]
# Explanation: One possible answer is [0,-3,9,-10,null,5], which represents the shown height balanced BST.
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        prev=None
        def rec(root):
            if root == None:
                return
            nonlocal prev
            rec(root.right)
            rec(root.left)

            root . right = prev
            root . left = None
            prev = root
        rec(root)
        
# T : O(N)
# S : O(1)

# Input: root = [1,2,5,3,4,null,6]
# Output: [1,null,2,null,3,null,4,null,5,null,6]
    def sumNumbers(self, root: Optional[TreeNode]) -> int:        
        def dfs(node, current):
            if node is None:
                return 0
            
            current = current * 10 + node.val
            if node.left is None and node.right is None:
                return current
            
            return dfs(node.left, current) + dfs(node.right, current)
            
        return dfs(root, 0)

# T : O(n)
# S : O(h)

# Input: root = [1,2,3]
# Output: 25
# Explanation:
# The root-to-leaf path 1->2 represents the number 12.
# The root-to-leaf path 1->3 represents the number 13.
# Therefore, sum = 12 + 13 = 25.
    def countNodes(self, root: Optional[TreeNode]) -> int:
        leftHeight, rightHeight = 0, 0
        cur = root
        while cur:
            cur = cur.left
            leftHeight += 1
        cur = root
        while cur:
            cur = cur.right
            rightHeight += 1
        if leftHeight == rightHeight: return (1 << rightHeight) - 1
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)        
    
# T : O(logN * logN)
# S : O(1)

#Given the root of a complete binary tree, return the number of the nodes in the tree.

#According to Wikipedia, every level, except possibly the last, is completely filled in a complete binary tree, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

#Design an algorithm that runs in less than O(n) time complexity.
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]
        graph = {i:[] for i in range(n)}
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        leaves = []
        for node in graph:
            if len(graph[node]) == 1:
                leaves.append(node)
        
        while len(graph) > 2:
            new_leaves = []
            for leaf in leaves:
                nei = graph[leaf].pop()
                del graph[leaf]
                graph[nei].remove(leaf)
                if len(graph[nei]) == 1:
                    new_leaves.append(nei)
            leaves = new_leaves
        
        return leaves 
# Input: n = 4, edges = [[1,0],[1,2],[1,3]]
# Output: [1]
# Explanation: As shown, the height of the tree is 1 when the root is the node with label 1 which is the only MHT.
