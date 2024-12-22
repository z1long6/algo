class Solution:
    # 暴力
    def minSubArrayLen1(self, target: int, nums: list[int]) -> int:
        lens = len(nums)
        for i in range(1, lens+1):
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
