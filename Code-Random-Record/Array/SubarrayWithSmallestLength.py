from collections import Counter

class Solution:
    # 暴力
    def minSubArrayLen1(self, target: int, nums: list[int]) -> int:
        lens = len(nums)
        for i in range(1, lens+1): # 窗口长度
            for j in range(lens-i+1):
                # sum = 0
                # 求和
                sum_result = sum(nums[j:j+i])
                # for k in range(j, j+i):
                #     sum += nums[k]
                if sum_result >= target:
                    return i
        return 0
    # slide window
    def minSubArrayLen2(self, target: int, nums: list[int]) -> int:
        nums_len = len(nums)
        left, right = 0, 0
        min_len = float('inf')
        cur_sum = 0 
        while right < nums_len:
            cur_sum += nums[right]
            while cur_sum >= target:
                min_len = min(min_len, right-left+1)
                cur_sum -= nums[left]
                left += 1
            right += 1

        return min_len if min_len != float('inf') else 0
    # LT.904.水果成篮子
    def totalFruit(self, fruits: list[int]) -> int:
        cnt_fruit = Counter()
        left, ans = 0, 0
        for right, x in enumerate(fruits):
            cnt_fruit[x] += 1 # 水果种类为x的数量
            while len(cnt_fruit) > 2:
                # 不满足条件,修改left使得窗口内再次满足条件
                cnt_fruit[fruits[left]] -= 1
                if cnt_fruit[fruits[left]] == 0:
                    cnt_fruit.pop(fruits[left]) # 删除fruit[left]
                left += 1
            ans = max(ans, right - left + 1)
        return ans
    # LT.713.乘积小于K的子数组
    def numSubarrayProductLessThanK(self, nums: list[int], k: int) -> int:
        if k <= 1: # nums[i] >= 1 
            return 0

        left = 0
        mul_num = 0
        result = 1
        for right, x in enumerate(nums):
            result *= x
            while result >= k: # 乘积结果大于k
                result //= nums[left]
                left += 1
            mul_num += right - left + 1 
        return mul_num
    # LT.3.无重复字符的最长子串
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 时间复杂度 O(n)
        # 空间复杂度 O(128) / O(1)

        counter_s = Counter()
        left, ans = 0, 0
        for right, c in enumerate(s):
            counter_s[c] += 1
            while max(counter_s.values()) > 1:
                counter_s[s[left]] -= 1
                left += 1
                if counter_s[s[left]] == 0:
                    counter_s.pop(s[left])
            ans = max(ans, right - left + 1)
        return ans
    # LT.76.最小覆盖子串
