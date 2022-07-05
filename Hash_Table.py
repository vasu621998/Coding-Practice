class MyHashSet:

    def __init__(self):
        self.array = [[] for _ in range(1000)]
        

    def add(self, key: int) -> None:
        subkey = key % 1000
        if not self.contains(key):
            self.array[subkey].append(key)

    def remove(self, key: int) -> None:
        subkey = key % 1000
        if self.contains(key):
            self.array[subkey].remove(key)        

    def contains(self, key: int) -> bool:
        subkey = key % 1000
        return key in self.array[subkey]


# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)
class MyHashMap:

    def __init__(self):
        self.l = [-1 for _ in range(1000001)]

    def put(self, key: int, value: int) -> None:
        self.l[key] = value

    def get(self, key: int) -> int:
        return self.l[key]

    def remove(self, key: int) -> None:
        self.l[key] = -1


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
