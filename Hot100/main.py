from collections import defaultdict
class Solution:
    '''
        finish hot100 in leetcode
    '''
    # LC.1.两数之和
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        mapping = defaultdict(int)
        for i, x in enumerate(nums):
            if target - x in mapping:
                return [i, mapping[target - x]]
            mapping[nums[i]] = i
        return []
    
    # LC.49.字母异位词分组
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        mapping = defaultdict(list)
        for i, s in enumerate(strs):
            mapping[''.join(sorted(s))].append(i)
        ans = []
        for index, value in mapping.items():
            temp = []
            for i, x in enumerate(value):
                temp.append(strs[x])
            ans.append(temp)
        return ans
    
    # LC.128.最长连续序列
    def longestConsecutive_0(self, nums: list[int]) -> int:
        if len(nums) == 0:
            return 0
        elif len(nums) == 1:
            return 1
        
        nums = list(set(nums))
        nums.sort()
        ans = 0
        for i in range(len(nums)-1):
            cnt = 1
            temp = nums[i]
            for j in range(i+1, len(nums)):
                if nums[j] - temp != 1:
                    break
                else:
                    cnt += 1
                    temp = nums[j]
                ans = max(ans, cnt)
        return ans
    
    # 事件复杂度限制在O(n), 不能对数组进行排序, 考虑使用哈希表
    def longestConsecutive_1(self, nums: list[int]) -> int:
        hashtable = set(nums)
        ans = 0
        for x in hashtable:

            if x - 1 in hashtable:
                continue

            y = x+1
            while y in hashtable:
                y += 1
            ans = max(ans, y-x)
            # 优化 hashtable不重复, 如存在长度大于等于m/2的子序列, 则该子序列的长度一定是最大的
            if ans * 2 >= len(hashtable):
                break
        return ans