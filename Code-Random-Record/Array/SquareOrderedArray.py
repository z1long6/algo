class Solution:
    # 暴力
    def sortedSquares1(self, nums: list[int]) -> list[int]:
        nums = sorted([num*num for num in nums])
        return nums
    # 双指针法
    def sortedSquares2(self, nums: list[int]) -> list[int]:
        i, j = 0, len(nums)-1
        k = len(nums)-1
        result = [int] * len(nums)
        while i <= j:
            if(nums[i]*nums[i] < nums[j]*nums[j]):
                result[k] = nums[j]*nums[j]
                j -= 1
            else:
                result[k] = nums[i]*nums[i]
                i += 1
            k -= 1
        return result