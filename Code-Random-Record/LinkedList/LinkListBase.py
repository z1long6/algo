from typing import Optional
from MyListNode import ListNode
'''
    链表的基础操作
'''
class Solutions:
    # LT.203.移除链表元素
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:  
        # 设置虚拟头节点
        new_head = ListNode(next=head)
        cur = new_head
        while cur.next:
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return new_head.next
    # LT.206.反转链表
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # if head == None or head.next == None:
        #     return head

        # dummy_head = ListNode(None, head) # 虚拟节点

        # # 使用栈采取迭代方法
        # stack_list = []
        # cur = dummy_head.next
        # while cur != None:
        #     stack_list.append(cur)
        #     cur = cur.next

        # stack_list.reverse()

        # for index in range(len(stack_list)):
        #     if index == len(stack_list)-1:
        #         stack_list[index].next = None
        #     else:
        #         stack_list[index].next = stack_list[index+1]

        # dummy_head.next = stack_list[0]

        # return dummy_head.next

        # 双指针迭代方法
        prev = None
        p = head
        while p != None:
            next = p.next # 暂存
            p.next = prev
            prev = p
            p = next
        return prev
    # LT.24.两两交换链表中的节点
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head == None or head.next == None:
            return head # 空链表或只有一个节点
        
        dummy_head = ListNode(None, head)
        node0 = dummy_head
        node1 = dummy_head.next


        while node1 != None and node1.next != None:
            node2 = node1.next
            node3 = node2.next

            node1.next = node3
            node2.next = node1
            node0.next = node2

            node0 = node1
            node1 = node3
        
        return dummy_head.next
    # LT.19.删除链表的倒数第N个节点
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # 手动获取链表长度
        # dummy_head = ListNode(None, head)
        # temp = head
        # len = 0
        # while temp != None:
        #     len += 1
        #     temp = temp.next
        
        # temp2 = dummy_head
        # for i in range(len-n):
        #     temp2 = temp2.next
        
        # temp2.next = temp2.next.next

        # return dummy_head.next

        # 前后指针，记录两个指针之间的差值
        left = right = dummy = ListNode(None, head)
        for i in range(n):
            right = right.next
        
        while right.next != None:
            right = right.next

            left = left.next
        
        left.next = left.next.next

        return dummy.next
    # LT.142.环形链表二
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # if head == None or head.next==None:
        #     return None

        # array = []
        # p = head
        # while p.next != None:
        #     array.append(p)
        #     for i in range(len(array)):
        #         if array[i] == p.next:
        #             return array[i]
        #     p = p.next

        # return None     
        # 快慢指针判断链表是否有环
        slow = fast = head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next

            if slow is fast: # there is a circle
                while slow is not head: # find the first circle node
                    slow = slow.next
                    head = head.next
                return slow
        return None