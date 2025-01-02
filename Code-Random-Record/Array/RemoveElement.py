class Solution:
    # LT.27.移除元素
    # 暴力遍历O(n2)
    def removeElement1(self, nums: list[int], val: int) -> int:
        nums_len = len(nums)
        i = 0
        while i < nums_len:
            if nums[i] == val:
                temp = nums[i]
                for j in range(i, nums_len-1):
                    nums[j] = nums[j+1]
                nums[len(nums)-1] = temp
                
                nums_len -= 1
                i -= 1
            i += 1
        return nums_len
    def removeElement2(self, nums: list[int], val: int) -> int:
        k = 0
        for item in nums:
            if item != val:
                nums[k] = item
                k += 1
        return k
    # 快慢指针
    def removeElement3(self, nums: list[int], val: int) -> int:
        slow, fast = 0, 0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        return slow
    # LT.26.删除有序数组中的重复项
    def removeDuplicates(self, nums: list[int]) -> int:
        # 快慢指针
        slow = 1
        for fast in range(len(nums)-1):
            if nums[fast] != nums[fast+1]:
                nums[slow] = nums[fast+1]
                slow += 1
        print(nums)
        return slow
    
        # 库函数
        # nums[:] = sorted(set(nums))
        # return len(nums)
    # LT.283.移动零
    def moveZeroes(self, nums: list[int]) -> None:
        
        pass