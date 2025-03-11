from collections import deque
from collections import Counter
import heapq
class Solution:
    # LT.20.有效的括号
    def isValid(self, s: str) -> bool:
        list = []
        for i, item in enumerate(s):
            if item in ['(', '[', '{']:
                list.append(item)
            elif item == ')':
                if len(list) == 0 or list.pop() != '(':
                    return False   
            elif item == ']':
                if len(list) == 0 or list.pop() != '[':
                    return False  
            elif item == '}':
                if len(list) == 0 or list.pop() != '{':
                    return False
        if len(list) != 0:
            return False  
        return True
    # LT.1047.删除字符串中的所有相邻重复项
    def removeDuplicates(self, s: str) -> str:
        stack = []
        for i, item in enumerate(s):
            if len(stack) > 0:
                if stack[len(stack)-1] == item:
                    stack.pop()
                    continue

            stack.append(item)

        return ''.join(stack)
    # LT.150.逆波兰表达式求值
    def evalRPN(self, tokens: list[str]) -> int:
        stack = []
        for index, item in enumerate(tokens):
            if item not in ['+', '-', '*', '/']:
                stack.append(item)
            else:
                b = stack.pop()
                a = stack.pop()
                if item == '+':
                    c = int(a) + int(b)
                elif item == '-':
                    c = int(a) - int(b)
                elif item == '*':
                    c = int(a) * int(b)
                elif item == '/':
                    c = int(a) / int(b)
                stack.append(c)
        return stack.pop()
    # LT.239.滑动窗口最大值
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        # O(n2) may be timeout in large k or too large nums list  // timeout
        # stack  = []
        # len_ = len(nums)
        # if len_ < k or len_ == k:
        #     return max(nums)

        # for i in range(len_ - k + 1):
        #     temp_list = []
        #     for j in range(i, i+k):
        #         temp_list.append(nums[j])
        #     stack.append(max(temp_list))

        # return stack
        # use Queue in one ergodic loop // timeout!
        # stack = []
        # len_ = len(nums)
        # if len_ < k or len_ == k:
        #     return [max(nums)]

        # myqueue = deque(maxlen=k)
        # for i in range(k):
        #     myqueue.append(nums[i])

        # stack.append(max(myqueue))

        # for i in range(k, len_):
        #     myqueue.popleft()
        #     myqueue.append(nums[i])
        #     stack.append(max(myqueue))
        
        # return stack
        
        # monotonic queue
        pass
    # LT.347.前K个高频元素
    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        # cnt = Counter(nums)
        # res = cnt.most_common(k)
        # ans = []
        # for i in range(len(res)):
        #     ans.append(res[i][0])
        # return ans
        #  heap
        cnt = Counter(nums)
        res = []
        for key, value in cnt.items():
            heapq.heappush(res, (value, key))
            if len(res) > k:
                heapq.heappop(res)
        
        res_ = []
        for i in range(len(res)-1, -1, -1):
            res_.append(heapq.heappop(res)[1])
        
        return res_

        