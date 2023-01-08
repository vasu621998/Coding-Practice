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
    def guessNumber(self, n: int) -> int:
        left, right = 0, n 
        while left <= right:
            mid = (left + right) // 2
            ans = guess(mid)
            if not ans:
                return mid
            elif ans == 1:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    
# T : O(logN)
# S : O(1)
# Input: n = 10, pick = 6
# Output: 6
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        output=[] 
        
        # sort the array in decreasing order of height 
        # within the same height group, you would sort it in increasing order of k
        # eg: Input : [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
        # after sorting: [[7,0],[7,1],[6,1],[5,0],[5,2],[4,4]]
        people.sort(key=lambda x: (-x[0], x[1]))                
        for a in people:
            # Now let's start the greedy here
            # We insert the entry in the output array based on the k value
            # k will act as a position within the array
            output.insert(a[1], a)
        
        return output  

# Time - O(nlogn + n * n) - We sort the array in O(nlogn) and the greedy algorithm loop takes O(n * n) because of array insert operation.
# Space - O(1) - If you exclude the output array.
# Input: people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
#Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
#Explanation:
#Person 0 has height 5 with no other people taller or the same height in front.
#Person 1 has height 7 with no other people taller or the same height in front.
#Person 2 has height 5 with two persons taller or the same height in front, which #is person 0 and 1.
#Person 3 has height 6 with one person taller or the same height in front, which #is person 1.
#Person 4 has height 4 with four people taller or the same height in front, which #are people 0, 1, 2, and 3.
#Person 5 has height 7 with one person taller or the same height in front, which #is person 1.
#Hence [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] is the reconstructed queue.
