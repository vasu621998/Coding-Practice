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
