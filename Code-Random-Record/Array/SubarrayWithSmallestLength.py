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
    def minSubArrayLen1(self, target: int, nums: list[int]) -> int:
        
        
        
        
        
        return 0