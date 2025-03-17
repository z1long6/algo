from collections import deque
class Solution:
    # LT.344.反转字符串
    def reverseString(self, s: list[str]) -> None:
        j = len(s) - 1
        for i in range(len(s) // 2):
            s[i], s[j] = s[j], s[i]
            j -= 1
    
    # LT.541.反转字符串2
    def reverseString(self, s: list[str], k: int) -> None:
        p = 0
        while p < len(s):
            p2 = p + k
            s = s[:p] + s[p:p2][::-1] + s[p2:]
            p += 2*k
        return s
    # LT.151.翻转字符串里的单词
    def reverseWords(self, s: str) -> str:
        return ' '.join(s.split(' ')[::-1])
    # LT.459.重复的子字符串
    def repeatedSubstringPattern(self, s: str) -> bool:
        # math
        return s in (s + s)[1:-1]    
    
    # LT.239.滑动窗口最大值
    # 滑动窗口框架
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        res = []
        queue_ = deque()
        for i, x in enumerate(nums):
            # 1.元素进入窗口
            while queue_ and nums[queue_[-1]] <= x:
                queue_.pop()
            queue_.append(i)
            # 2.元素离开窗口
            if i-queue_[0] >= k:
                queue_.popleft()

            # 3.记录答案
            if i >= k-1:
                res.append(nums[queue_[0]])
        return res