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
        left, right = 0, len(nums)-1
        while(left <= right):
            mid = (left + right) // 2
            if nums[mid] == target: # 找到该数
                temp_left_mid = mid-1
                temp_right_mid = mid+1

                while(temp_left_mid >= 0 and nums[temp_left_mid] == target):
                    temp_left_mid -= 1
                while(temp_right_mid < len(nums) and nums[temp_right_mid] == target):
                    temp_right_mid += 1
                
                if(temp_left_mid == -1 and temp_right_mid != len(nums)):
                    return [0, temp_right_mid]
                elif(temp_left_mid != -1 and temp_right_mid == len(nums)):
                    return [temp_left_mid, len(nums)-1]
                elif(temp_left_mid == -1 and temp_right_mid == len(nums)):
                    return [0, len(nums)-1]
                else:
                    return [temp_left_mid+1, temp_right_mid-1]

            elif nums[mid] > target:
                right = mid-1
            else:
                left = mid+1

        # 未找到该数
        return [-1, -1]
    # LT.69.x的平方根
    def mySqrt(self, x: int) -> int:
        # 暴力遍历
        # temp = 0
        # while temp * temp <= x:
        #     temp += 1
        # return temp - 1 
        # 二分法
        left, right = 0, min(x+1, 46341)
        while left + 1 < right: # 开区间不为空
            mid = (left + right) // 2
            if mid * mid <= x:
                left = mid
            else:
                right = mid
        return left
    # LT.367.有效完全平方数
    def isPerfectSquare(self, num: int) -> bool:
        left, rigth = 1, num // 2 + 1
        while left <= rigth:
            mid = (left + rigth) // 2
            temp = mid * mid
            if temp == num:
                return True
            elif temp < num:
                left = mid + 1
            else:
                rigth = mid - 1
        return False
    