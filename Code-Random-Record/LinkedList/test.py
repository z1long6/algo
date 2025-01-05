
import LinkListBase
from MyListNode import ListNode

# 将nums 转为LinkList
def nums2LinkList(nums: list[int]) -> ListNode:
    if len(nums) == 0:
        return None
    
    head = ListNode(val = nums[0], next = None)
    p = head
    for i in range(1, len(nums)):
        temp = ListNode(val = nums[i], next = None)
        p.next = temp
        p = temp
    
    return head

# 遍历输出LinkList
def printLinkList(head: ListNode):
    while head != None:
        print(head.val)
        head = head.next

if __name__ == '__main__':

    nums = [1,2,6,3,4,5,6]
    val = 6

    head = nums2LinkList(nums)
    # printLinkList(head)
    # 移除链表元素
    printLinkList(LinkListBase.Solutions.removeElements(None, head, val))