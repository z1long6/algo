# 闭区间写法
# 第一个大于等于target的元素的数组下标
# 数组元素按照递增有序排序
def lower_bound(nums: list, target: int) -> int:
    left, right = 0, len(nums)-1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    # 循环结束时 left = right + 1
    return left

import math
from bisect import bisect_right, bisect_left
from TreeNode import TreeNode
from typing import Optional

class Solution:
    """
        题单: 二分算法
    """
    # LC.34.在排序数组中寻找元素的第一个和最后一个位置
    def searchRange(self, nums: list[int], target: int) -> list[int]:
        start = lower_bound(nums, target)
        if nums[start] != target or start == len(nums):
            return [-1, -1]
        end = lower_bound(nums, target+1) - 1
        return [start, end]
    
    # LC.35.搜索插入位置
    def searchInsert(self, nums: list[int], target: int) -> int:
        return lower_bound(nums, target)

    # LC.2300.咒语和药水的成功对数
    def successfulPairs(self, spells: list[int], potions: list[int], success: int) -> list[int]:
        ans = []
        n = len(potions)
        potions.sort()
        for i, x in enumerate(spells):
            temp = math.ceil(success / x)
            ans.append(n - lower_bound(potions, temp))
        return ans
    
    # LC.2389.和有限的最长子序列
    def answerQueries(self, nums: list[int], queries: list[int]) -> list[int]:
        ans = []
        nums.sort()
        # 枚举queries中的每一个数字, 找到第一个 "小于等于queries[i]" 的数字, 符合条件的子序列在i的左侧
        for i, x in enumerate(queries):
            j = lower_bound(nums, x+1) - 1 # 小于等于x的位置下标
            # 贪心从最小的数字开始枚举, 直到不满足条件
            temp = 0
            sum_ = 0
            while temp <= j and sum_ <= x:
                sum_ += nums[temp]
                if sum_ > x:
                    break
                temp += 1

            ans.append(temp)
        return ans
    
    # LC.1170.比较字符串最小字母出现频次
    def numSmallerByFrequency(self, queries: list[str], words: list[str]) -> list[int]:
        def f(s):
            s = sorted(s)
            # 长度
            return bisect_right(s, s[0])
        
        q_n, w_n = [], []
        for i in queries:
            q_n.append(f(i))

        for i in words:
            w_n.append(f(i))

        w_n.sort()
        n = len(words)

        # 长度大于 i+1 的第一个数字下标位置j, 满足条件的长度即为 n-j
        return [n - bisect_left(w_n, i + 1) for i in q_n]
    
    # LC.2476.二叉搜索树最近节点查询
    '''
        中序遍历二叉搜索树, 得到非递减数组
    '''
    def closestNodes(self, root: Optional[TreeNode], queries: list[int]) -> list[list[int]]:
       
        def inOrderReverse(root: Optional[TreeNode]) -> Optional[list[int]]:
            order = []
            if root == None:
                return None
            inOrderReverse(root.left)
            order.append(root.val)
            inOrderReverse(root.right)
            return order
        order = inOrderReverse(root)
        ans = []
        assert order is not None # 长度至少为2
        for _, x in enumerate(queries):
            # 大于等于x的最小值
            j = bisect_left(order, x)
            max_ = order[j] if j < len(order) else -1
            if j == len(order) or order[j] != x: # order[j] < x
                j -= 1
            min_ = order[j] if j >= 0 else -1
            ans.append([min_, max_])
        return ans
    
    # LC.1283.使结果不超过阈值的最小常数
    def smallestDivisor(self, nums: list[int], threshold: int) -> int:

        # 检测数值是否合法
        def check(temp):
            return sum(math.ceil(x / temp) for x in nums) <= threshold
        i, j = 1, max(nums)-1
        while i < j:
            mid = i + (j - i) // 2
            if check(mid):
                j = mid -1
            else:
                i = mid + 1
        return i