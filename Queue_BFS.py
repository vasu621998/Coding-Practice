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
    def openLock(self, deadends: List[str], target: str) -> int:
        if "0000" in deadends:
            return -1
        
        def children(lock):
            res = []
            for i in range(4):
                digit = str((int(lock[i]) + 1) % 10)
                res.append(lock[:i] + digit + lock[i+1:])
                digit = str((int(lock[i]) - 1 + 10) % 10)
                res.append(lock[:i] + digit + lock[i+1:])
            return res
        
        q= deque()
        q.append(["0000", 0])
        visit = set(deadends)
        
        
        while q:
            lock, turns = q.popleft()
            if lock == target:
                return turns
            for child in children(lock):
                if child not in visit:
                    visit.add(child)
                    q.append([child, turns + 1])
        return -1
    def numSquares(self, n: int) -> int:
        arr = [n] * (n + 1)
        arr[0] = 0
        
        for target in range(1, n + 1):
            for s in range(1, target + 1):
                square = s * s
                if target - square < 0:
                    break
                arr[target] = min(arr[target] , 1 + arr[target - square])
                
        return arr[n]
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        h = len(mat)
        w = len(mat[0])
        q = []
        
        for i in range(h):
            for j in range(w):
                if mat[i][j] == 0:
                    q.append((i,j))
                else:
                    mat[i][j] = "#"
            
        
        for row,col in q:
            for dx, dy in (1,0),(-1,0),(0,1),(0,-1):
                nr = row + dx
                nc = col + dy
                
                if 0 <= nr < h and 0 <= nc < w and mat[nr][nc] == "#":
                    mat[nr][nc] = mat[row][col] + 1
                    q.append((nr,nc))
                    
        return mat
    def pushDominoes(self, dominoes: str) -> str:
        dominoes = list(dominoes)
        active = set([i for i, d in enumerate(dominoes) if d != '.'])
        while active:
            nxt_active = set()
            for i in active:
                if dominoes[i] == 'L':
                    if i > 0:
                        if dominoes[i-1] == '.':
                            dominoes[i-1] = 'L'
                            nxt_active.add(i-1)
                        elif dominoes[i-1] == 'R':
                            if i-1 not in active:
                                dominoes[i-1] = '.'
                                nxt_active.remove(i-1)
                else:
                    if i < len(dominoes) - 1:
                        if dominoes[i+1] == '.':
                            dominoes[i+1] = 'R'
                            nxt_active.add(i+1)
                        elif dominoes[i+1] == 'L':
                            if i+1 not in active:
                                dominoes[i+1] = '.'
                                nxt_active.remove(i+1)
            active = nxt_active
        return "".join(dominoes) 
