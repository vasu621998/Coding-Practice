#LinkedList Questions of Leetcode
class Node(object):
    def __init__(self,val):
        self.val = val
        self.next = None

class MyLinkedList:

    def __init__(self):
        self.head = None
        self.size = 0
        

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        if self.head is None:
            return -1
        curr = self.head
        for i in range(index):
            curr = curr.next
        return curr.val
        

    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0, val)
        

    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return
        node = Node(val)
        if index == 0:
            node.next = self.head
            self.head = node
        else:
            curr = self.head
            if curr is None:
                self.head = node
            else:
                for i in range(index - 1):
                    curr = curr.next
                node.next = curr.next
                curr.next = node
        self.size += 1
        

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        curr = self.head
        if index == 0:
            self.head = curr.next
        else:
            for i in range(index - 1):
                curr = curr.next
            curr.next = curr.next.next
        self.size -= 1


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# Definition for singly-linked list.

## Linked List Problems : 

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
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

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        l1,l2 = headA, headB
        while l1 != l2:
            l1 = l1.next if l1 else headB
    
            l2 = l2.next if l2 else headA
        
        return l1
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0,head)
        left = dummy
        right = head
        
        while n > 0 :
            right = right.next
            n -= 1
            
        while right :
            left = left.next
            right = right.next
            
        #delete
        
        left.next = left.next.next
        return dummy.next
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev
     def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy = ListNode(val = 0,next=head)
        
        prev, curr = dummy, head
        
        while curr:
            temp = curr.next
            if curr.val == val:
                prev.next = temp
            else:
                prev = curr
            
            curr = temp
        return dummy.next
