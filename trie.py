class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        m = {}
        for i, word in enumerate(words):
            m[word] = i
        
        result = set()
        for i, word in enumerate(words):
            n, rev_word = len(word), word[::-1]
            prefix, suffix = word, rev_word
            
            for j in range(n+1):
                if prefix == suffix:
                    key = rev_word[:j]
                    if key in m and m[key] != i:
                        result.add((m[key], i))
                
                if j == n:
                    break
                
                prefix = prefix[:-1]
                suffix = suffix[1:]
            
            # print('pre', i, result)
            
            prefix, suffix = '', ''
            for j in range(n+1):
                if prefix == suffix:
                    if prefix == suffix:
                        key = rev_word[j:]
                        if key in m and m[key] != i:
                            result.add((i, m[key]))
                
                if j == n:
                    break
                
                prefix = word[n-j-1] + prefix
                suffix = suffix + rev_word[j]
            
            # print('post', i, result)
        
        return list(result)        
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board or not words: return []
        boardc = collections.Counter(sum(board,[]))
        words = [word for word in words if collections.Counter(word) <= boardc]
        trie = {}
        for word in words:
            node = trie
            for c in reversed(word):
                node = node.setdefault(c, {})
            node['$'] = word
            
        def find(i,j,node):
            if '$' in node: out.append(node.pop('$'))
            if not node: return
            tmp, board[i][j] = board[i][j], '#'
            
            for dx,dy in [(i+1, j),(i-1,j), (i,j+1), (i,j-1)]:
                if 0<=dx<m and 0<=dy<n and board[dx][dy] in node:
                    find(dx,dy,node[board[dx][dy]])
                    if not node[board[dx][dy]]: node.pop(board[dx][dy])
            board[i][j] = tmp
            
            return 0
                    
        m, n = len(board), len(board[0])
        out = []
        for i in range(m):
            for j in range(n):
                if board[i][j] in trie:
                    find(i,j,trie[board[i][j]])
                    if not trie[board[i][j]]: trie.pop(board[i][j])
                if len(words) == len(out): return out
        return out     
class Trie:
    def __init__(self):
        self.root = {}
    def insert(self, num):
        cur = self.root
        for j in range(31, -1, -1):
            tmp_bit = num & (1 << j)
            c = 1 if tmp_bit else 0
            if c not in cur:
                cur[c] = {}
            cur = cur[c]

class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        trie = Trie()
        for num in nums:
            trie.insert(num)
        #if the higher bit is geater, the greater is the number
        res = 0
        for num in nums:
            cur = trie.root
            tmpval = 0
            for j in range(31,-1,-1):
                tmp_bit = num & 1 << j
                c = 0 if tmp_bit else 1
                if c in cur:
                    cur = cur[c]
                    tmpval += 1 << j
                else:
                    cur = cur[not c]
            res = max(res, tmpval)
        return res        
