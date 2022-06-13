class MinStack:

    def __init__(self):
        self.stack =[]
        self.minStack = []
        

    def push(self, val: int) -> None:
        self.stack.append(val)
        val = min(val, self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.minStack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minStack[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        closeToopen = { ")" : "(", "}" : "{", "]" : "["}
        
        for c in s:
            if c in closeToopen:
                if stack and stack[-1] == closeToopen[c]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(c)
        
        return True if not stack else False
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = []
        
        for i,t in enumerate(temperatures):
            while stack and t > stack[-1][0]:
                stackT, stackI = stack.pop()
                res[stackI] = ( i - stackI)
            stack.append([t, i])
            
        return res
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for c in tokens:
            if c == "+":
                stack.append(stack.pop() + stack.pop())
            elif c == "-":
                a, b = stack.pop(), stack.pop()
                stack.append(b-a)
            elif c == "*":
                stack.append(stack.pop() * stack.pop())
            elif c == "/":
                a, b = stack.pop(), stack.pop()
                stack.append(int(b/a))
                
            else:
                stack.append(int(c))
                    
        return stack[0]
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
    def cloneGraph(self, node: 'Node') -> 'Node':
        oldtoNew = {}
        
        def dfs(node):
            if node in oldtoNew:
                return oldtoNew[node]
            
            copy = Node(node.val)
            oldtoNew[node] = copy
            
            for n in node.neighbors:
                copy.neighbors.append(dfs(n))
            return copy
        
        return dfs(node) if node else None
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        dp = {} #(index,total)
        
        def backtrack(index, total):
            if index == len(nums):
                return 1 if total==target else 0
            if (index,total) in dp:
                return dp[(index,total)]
            dp[(index,total)] = (backtrack(index+1, total + nums[index]) +
                                 backtrack(index+1, total - nums[index]) )
            
            return dp[(index, total)]
        return backtrack(0,0)
    def decodeString(self, s: str) -> str:
        stack = []
        for ch in s:
            if ch == "]" and stack:
                el = ""
                while stack and not el.startswith("["):
                    el = stack.pop() + el
                while stack and stack[-1].isdigit():
                    el = stack.pop() + el

                num, el = el.strip("]").split("[")
                ch = el * int(num)

            stack.append(ch)
        return "".join(stack)
