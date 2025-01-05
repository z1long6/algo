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