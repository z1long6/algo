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
        if head == None or head.next == None:
            return head

        dummy_head = ListNode(None, head) # 虚拟节点

        # 使用栈采取迭代方法
        stack_list = []
        cur = dummy_head.next
        while cur != None:
            stack_list.append(cur)
            cur = cur.next
            
        stack_list.reverse()

        for index in range(len(stack_list)):
            if index == len(stack_list)-1:
                stack_list[index].next = None
            else:
                stack_list[index].next = stack_list[index+1]

        dummy_head.next = stack_list[0]

        return dummy_head.next
            
