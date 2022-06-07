class ListNode:
    def __init__(self, val, next, prev):
        self.val,self.next,self.prev = val,next,prev

class MyCircularQueue:

    def __init__(self, k: int):
        self.space = k
        self.left = ListNode(0, None, None)
        self.right = ListNode(0, None, self.left)
        self.left.next = self.right

    def enQueue(self, value: int) -> bool:
        if self.isFull(): return False
        curr = ListNode(value, self.right, self.right.prev)
        self.right.prev.next = curr
        self.right.prev = curr
        self.space -= 1
        return True

    def deQueue(self) -> bool:
        if self.isEmpty(): return False
        self.left.next = self.left.next.next
        self.left.next.prev = self.left
        self.space += 1
        return True

    def Front(self) -> int:
        if self.isEmpty(): return -1
        return self.left.next.val
        

    def Rear(self) -> int:
        if self.isEmpty(): return -1
        return self.right.prev.val
        

    def isEmpty(self) -> bool:
        return self.left.next == self.right
    

    def isFull(self) -> bool:
        return self.space == 0


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        
        island = 0
        visit = set()
        rows, cols = len(grid), len(grid[0])
        
        def bfs(r,c):
            q = collections.deque()
            visit.add((r,c))
            q.append((r,c))
            while q:
                row, col = q.popleft()
                directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                for dr, dc in directions:
                    r  = row + dr
                    c = col + dc
                    if(r in range(rows) and
                      c in range(cols) and
                      grid[r][c] == "1" and
                      (r,c) not in visit):
                        visit.add((r,c))
                        q.append((r,c))
                    
            
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r,c) not in visit:
                    bfs(r,c)
                    island += 1
        return island
