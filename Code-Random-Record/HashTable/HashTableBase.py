from collections import Counter
class Solution:
    # LT.242.有效的字母异位词
    def isAnagram(self, s: str, t: str) -> bool:
        counter1 = Counter(s)
        counter2 = Counter(t)
        if counter1 == counter2:
            return True
        else:
            return False
    # LT.349.两个数组之间的交集
    def intersection(self, nums1: list[int], nums2: list[int]) -> list[int]:
        counter_num1 = Counter(nums1)
        counter_num2 = Counter(nums2)
        result = set.intersection(set(counter_num1.keys()), set(counter_num2.keys()))
        return list(result)
    # LT.202.快乐数
    def isHappy(self, n: int) -> bool:
        def get_next(n):
            total_sum = 0
            while n > 0:
                n, digit = divmod(n, 10)
                total_sum += digit ** 2
            return total_sum
        slow = n
        fast = get_next(n)
        while fast != 1 and fast != slow:
            slow = get_next(slow)
            fast = get_next(get_next(fast))
        return fast == 1
    
    def search(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums)-1
        while(left <= right):
            mid = (left + right) // 2
            if nums[mid] == target:
                return nums[mid]
            elif nums[mid] > target:
                right = mid-1
            else:
                left = mid+1
        return None
    
    # LT.1.两数之和
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        # 优化搜索
        # my_nums = sorted(nums)
        # tempa = tempb = None
        # for i in range(len(my_nums)-1):
        #     right = Solution.search(None, my_nums[i+1:len(my_nums)], target-my_nums[i])
        #     if right != None:
        #         for j in range(len(nums)):
        #             if my_nums[i] == nums[j] and tempa == None:
        #                 tempa = j

        #             if right == nums[j] and j != tempa:
        #                 tempb = j
        
        # return [tempa, tempb]
        # 哈希
        record = dict()
        for index, item in enumerate(nums):
            if target - item in record:
                return [index, record[target - item]]
            record[item] = index
        return []
     # LT.454.四数相加
    def fourSumCount(self, nums1: list[int], nums2: list[int], nums3: list[int], nums4: list[int]) -> int:
        record1 = record2 = dict()
        for index1, item1 in enumerate(nums1):
            for index2, item2 in enumerate(nums2):
                if item1 + item2 in record1:
                    record1[item1+item2] += 1
                else:
                    record1[item1+item2] = 1
        
        res = 0
        for index1, item1 in enumerate(nums3):
            for index2, item2 in enumerate(nums4):
                if -(item1+item2) in record1:
                    res += record1[-(item1+item2)]
        return res
    
    # LT.383.赎金信
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        cnt1 = Counter(ransomNote)
        cnt2 = Counter(magazine)
        for key in cnt1.keys():
            if not ((key in cnt2.keys()) and (cnt1[key] < cnt2[key] or cnt1[key] == cnt2[key])):
                return False
        return True
    
    # LT.15.三数之和
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        # 双指针
        nums.sort()
        res = []

        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left, right = i+1, len(nums)-1
            while left < right:
                if nums[i] + nums[left] + nums[right] == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif nums[i] + nums[left] + nums[right] < 0:
                    left += 1
                else:
                    right -= 1
        return res