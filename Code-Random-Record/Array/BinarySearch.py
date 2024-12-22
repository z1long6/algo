class Solution:
    # 二分查找
    def search(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums)-1
        while(left <= right):
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid-1
            else:
                left = mid+1
        return -1
    
    # LT.35.搜索插入位置
    def searchInsert(self, nums: list[int], target: int) -> int:
        # binary search
        left, right = 0, len(nums)-1
        while(left <= right):
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid # find target
            elif nums[mid] > target:
                right = mid-1
            else:
                left = mid+1

        # do not find target
        if left == len(nums):
            return len(nums)
        elif right == -1:
            return 0
        else:
            return left
        
    # LT.34.在排序数组中查找元素的第一个和最后一个位置
    def searchRange(self, nums: list[int], target: int) -> list[int]:
        return 0 