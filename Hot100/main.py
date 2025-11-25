from collections import defaultdict, Counter, deque
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
    
    # LC.11.盛最多水的容器
    def maxArea(self, height: list[int]) -> int:
        ans = 0
        left, right = 0, len(height)-1
        while left < right:
            a, b = height[left], height[right]
            if a <= b:
                ans = max(ans, a*(right-left))
                # height[left]无法在右侧垂线中找到比ans更大的面积, 所以直接更新left += 1, 寻找更大值
                # ans > 任意一个包含height[left]的面积组合
                left += 1
            else:
                ans = max(ans, b*(right-left))
                right -= 1
        return ans   
    
    # LC.15.三数之和
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        ans = []
        nums.sort()
        for i in range(len(nums)-2):
            j, k = i+1, len(nums)-1
            # 去重
            if i > 0 and nums[i] == nums[i-1]:
                continue
            while j < k:
                s = nums[i] + nums[j] + nums[k]
                if s < 0:
                    j += 1
                elif s > 0:
                    k -= 1
                else:
                    ans.append([nums[i], nums[j], nums[k]])
                    j += 1
                    while j < k and nums[j] == nums[j-1]:
                        j += 1
                    k -= 1
                    while k > j and nums[k] == nums[k+1]:
                        k -= 1
        return ans
    
    # LC.42.接雨水
    # 数形结合, 横着计算面积
    def trap_0(self, height: list[int]) -> int:
        left, right = 0, len(height)-1
        s = 0
        for i in range(1, max(height)+1):
            while left < i:
                left += 1
            while right < i:
                right -= 1
            s += right - left + 1
        return s - sum(height)

    # 单调栈
    def trap_1(self, height: list[int]) -> int:
        ans = 0
        st = []
        for i, x in enumerate(height):
            while st and height[st[-1]] < x:
                bottom_h = height[st.pop()]
                if not st:
                    break
                left = st[-1]
                h = min(height[left], x) - bottom_h
                ans += h * (i - left -1)
            st.append(i)
        return ans
    
    # LC.438.字母异位词
    def findAnagrams(self, s: str, p: str) -> list[int]:
        ans = []
        m = len(p)
        templist = deque()
        for i, x in enumerate(s):
            templist.append(x)

            left = i - m + 1            

            if left < 0:
                continue

            if len(templist) == m and str(sorted(templist)) == str(sorted(list(p))):
                ans.append(left)
            
            if left >= 0:
                templist.popleft()

        return ans
    
    # LC.560.和为K的子数组
    # 数组内存在负数, 无法始终保持窗口内元素的单调性, 无法使用滑动窗口
    # 要计算连续子数组的和, 考虑使用前缀和
    # s[j] - s[i] = k
    # 枚举j, 计算 i < j时有多少个i符合条件
    # s[i] = s[j] - k
    def subarraySum(self, nums: list[int], k: int) -> int:
        ans = 0
        # 计算前缀和
        s = [0] * (len(nums)+1)
        for i, x in enumerate(nums):
            s[i+1] += s[i] + x
        cnt = defaultdict(int)
        for sj in s:
            ans += cnt[sj-k]
            cnt[sj] += 1
        return ans
    
    """
        单调队列
        队头到队尾呈现单调性, 本题中是单调递减
    """
    # LC.239.滑动窗口最大值
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        ans = []
        q = deque()
        for i, x in enumerate(nums):
            # 1. 元素加入队列, 保证队列单调递减
            while q and nums[q[-1]] <= x:
                q.pop()
            q.append(i)
            # 2. 队首元素离开队列, 此时滑动窗口位置不再包含队首元素, 寻找新的队首元素
            if i - q[0] + 1 > k:
                q.popleft()
            # 3. 更新
            if i >= k - 1:
                ans.append(nums[q[0]])
        return ans
    
    # LC.189.轮转数组
    # 使用O(1)的空间复杂度进行原地反转
    def rotate(self, nums: list[int], k: int) -> None:
        def my_reverse(i: int, j: int):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        n = len(nums)
        k = k % n
        my_reverse(0, n-1)
        my_reverse(0, k-1)
        my_reverse(k, n-1)

    # 除自身之外的数组乘积
    # 和前缀和的区别在于不需要计算下标为0和n-1的两个数字
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        n = len(nums)
        preffix, suffix = [1] * n, [1] * n
        # 构建前缀
        for i in range(1, n):
            preffix[i] = preffix[i-1] * nums[i-1]

        # 构建后缀
        for j in range(n-2, -1, -1):
            suffix[j] = suffix[j+1] * nums[j+1]
        
        return [p * s for (p, s) in zip(preffix, suffix)]