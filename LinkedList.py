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
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next or not head.next.next:
            return head
        
        oddList = curr = head
        evenList = evenhead = head.next
        i = 1
        while curr:
            if i > 2 and  i % 2 != 0:                
                oddList.next = curr
                oddList = oddList.next
            elif i > 2 and  i % 2 == 0:
                evenList.next = curr
                evenList = evenList.next                
            curr = curr.next
            i += 1
            
        evenList.next = None
        oddList.next = evenhead
        
        return head
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        slow,fast = head, head
        
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            
        prev = None
        while slow:
            temp = slow.next
            slow.next = prev
            prev = slow
            slow = temp
        
        left,right = head, prev
        while right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        return True
class DoublyLinkedList:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.head = None
        self.tail = None
        self.length = 0
        

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        
        if index < 0 or index >= self.length:
            return -1
        
        cur = self.head
        
        while index != 0:
            
            cur = cur.next
            index -= 1
            
        return cur.val
            

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        
        new_node = ListNode( val )
        
        new_node.next = self.head
        
        if self.head:
            self.head.prev = new_node
        
        self.head = new_node
               
        self.length += 1
        
        if self.length == 1:
            self.tail = new_node

        ### trace and debug
        #self.print_linked_list()
        
        
    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        
        new_node = ListNode( val )
        
        new_node.prev = self.tail
        
        if self.tail:
            self.tail.next = new_node
        
        self.tail = new_node
        
        self.length += 1
        
        if self.length == 1:
            self.head = new_node

        ### trace and debug
        #self.print_linked_list()            
        
        
    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        
        if index < 0 or index > self.length:
            return
        
        elif index == 0:
            self.addAtHead( val )
        
        elif index == self.length:
            self.addAtTail( val )
            
        else:
                
            cur = self.head
            while index-1 != 0:

                cur = cur.next
                index -= 1

            new_node = ListNode( val )

            new_node.next = cur.next
            cur.next.prev = new_node

            cur.next = new_node
            new_node.prev = cur
            
            self.length += 1

        ### trace and debug
        #self.print_linked_list()
        

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        
        if index < 0 or index >= self.length:
            return
        
        elif index == 0:
            
            if self.head.next:
                self.head.next.prev = None
                
            self.head = self.head.next
            
            self.length -= 1
            
            if self.length == 0:
                self.tail = None
        
        elif index == self.length-1:
            
            if self.tail.prev:
                self.tail.prev.next = None
            
            self.tail = self.tail.prev
            
            self.length -= 1
            
            if self.length == 0:
                self.head = None
            
        else:
                
            cur = self.head
            while index-1 != 0:

                cur = cur.next
                index -= 1

            cur.next = cur.next.next
            cur.next.prev = cur
            
            self.length -= 1

        ### trace and debug
        #self.print_linked_list()            
            
            
    def print_linked_list(self):
        
        cur = self.head
        
        while cur:
            print( f' {cur.val} -> ', end = '')
            cur = cur.next
        
        print('\n')
        
        return
                
# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        tail = dummy
        
        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        
        if list1:
            tail.next = list1
        elif list2:
            tail.next = list2
        
        return dummy.next
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        curr = dummy
        carry = 0
        
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            
            val = v1 + v2 + carry
            
            carry = val // 10
            val = val % 10
            
            curr.next = ListNode(val)
            
            curr = curr.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            
        return dummy.next
    def flatten(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if head != None: self.myfunc(head)
        return head
        
    def myfunc(self, head):
        curr,tail = head, head
        while curr != None:
            child = curr.child
            temp = curr.next
            if child != None:
                _tail = self.myfunc(child)
                _tail.next  = temp
                if temp != None:
                    temp.prev = _tail
                curr.next = child
                child.prev = curr
                curr.child = None
                curr = _tail
            else:
                curr = temp
            if curr != None:
                tail = curr
        return tail
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        
        myHashMap = {None : None}
        
        curr = head
        
        while curr:
            copy = Node(curr.val)
            myHashMap[curr] = copy
            curr = curr.next
            
        curr = head    
        while curr:
            copy = myHashMap[curr]
            copy.next = myHashMap[curr.next]
            copy.random = myHashMap[curr.random]
            curr = curr.next
        return myHashMap[head]
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head:
            return head  
        
        length, tail = 1, head
        while tail.next:
            tail = tail.next
            length += 1
        
        k = k % length
        if k == 0 :
            return head
        
        curr = head
        
        for i in range(length - k - 1):
            curr = curr.next
        newHead = curr.next
        curr.next = None
        tail.next = head
        
        return newHead
