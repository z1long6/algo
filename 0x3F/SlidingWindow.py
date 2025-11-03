from collections import Counter, deque, defaultdict
import math
class Solution:
    # LC.1456.定长子串中元音的最大数目
    # 法1: 暴力
    def maxVowels_0(self, s: str, k: int) -> int:
        my_list = ['a', 'e', 'i', 'o', 'u']
        ans = 0
        for i in range(len(s)-k+1):
            temp_str = s[i:i+k]
            cnt = Counter(temp_str)
            temp = 0
            for key, value in cnt.items():
                if key in my_list:
                    temp += int(value)
            ans = max(ans, temp)
                
        return ans
    
    # 法2: 
    def maxVowels_1(self, s: str, k: int) -> int:
        ans = 0
        temp = 0
        for i, x in enumerate(s):

            if x in 'aeiou':
                temp += 1
            
            left = i-k+1
            
            if left < 0:
                continue
            # 本次窗口计算完成，更新ans
            ans = max(ans, temp)
            # 滑动窗口向右移动
            if s[left] in 'aeiou':
                temp -= 1

        return ans
    
    # LC.643.子数组最大平均数1
    def findMaxAverage(self, nums: list[int], k: int) -> float:
        ans = -float('inf')
        temp = 0
        for i, x in enumerate(nums):
            temp += x

            left = i - k + 1
            if left < 0:
                continue

            ans = max(ans, temp / k)
            temp -= nums[left]
        return ans
    
    # LC.1343.大小为K且平均值大于等于阈值的子数组数目
    def numOfSubarrays(self, arr: list[int], k: int, threshold: int) -> int:
        ans = 0
        temp = 0
        for i, x in enumerate(arr):
            temp += x

            left = i - k + 1
            if left < 0:
                continue

            if temp / k >= threshold:
                ans += 1

            temp -= arr[left]
        return ans
    
    # LC.2090.半径为k的子数组平均值
    def getAverages(self, nums: list[int], k: int) -> list[int]:
        ans = [0] * len(nums)
        temp = 0
        for i, x in enumerate(nums):
            left, right = i - k, i + k
            # 右半径截止
            if right > len(nums)-1:
                for index in range(i, len(nums)):
                    ans[index] = -1
                break

            temp += nums[right]

            if left < 0:
                ans[i] = -1
                continue
            else:
                if left == 0:
                    temp = temp + sum(nums[0:i])
                    ans[i] = temp // (2*k+1)
                else:
                    ans[i] = temp // (2*k+1)

            temp -= nums[left]
                
        return ans
    
    # LC.2379.得到K个黑块的最少涂色次数
    def minimumRecolors(self, blocks: str, k: int) -> int:
        ans = temp =  0
        for i, x in enumerate(blocks):
            if x == 'B':
                temp += 1
            left = i - k + 1
            if left < 0:
                continue
            ans = max(ans, temp)
            if ans == k:
                break

            if blocks[left] == 'B':
                temp -= 1
            

        return k - ans
    
    # LC.2841.几乎唯一子数组的最大和
    def maxSum(self, nums: list[int], m: int, k: int) -> int:
        ans = temp = 0
        cnt = Counter()
        for i, x in enumerate(nums):
            temp += x
            cnt[x] += 1

            left = i - k + 1
            if left < 0:
                continue

            if len(cnt) >= m:
                ans = max(ans, temp)

            temp -= nums[left]
            cnt[nums[left]] -= 1
            if cnt[nums[left]] == 0:
                del cnt[nums[left]]

        return ans
    
        # ans = temp = 0
        
        # for i, x in enumerate(nums):
        #     temp += x
        #     left = i - k + 1
        #     if left < 0:
        #         continue
        #     cnt = Counter(nums[left:i+1])
        #     if len(cnt) >= m:
        #         ans = max(ans, temp)
        #     temp -= nums[left]

        # return ans

    # LC.2461.长度为K的子数组中的最大和
    def maximumSubarraySum(self, nums: list[int], k: int) -> int:
        ans = temp = 0
        cnt = Counter()
        for i, x in enumerate(nums):
            temp += x
            cnt[x] += 1

            left = i-k+1
            if left < 0:
                continue

            if len(cnt) == k:
                ans = max(ans, temp)
            
            temp -= nums[left]
            cnt[nums[left]] -= 1

        return ans

    # 双指针
    # LC.1423.可获得的最大点数
    def maxScore(self, cardPoints: list[int], k: int) -> int:
        m = len(cardPoints) - k
        ans = sum_ = sum(cardPoints[:m])
        for i in range(m, len(cardPoints)):
            sum_ += cardPoints[i] - cardPoints[i - m]
            ans = min(ans, sum_)

        return sum(cardPoints) - ans
    
    # LC.3679.使库存平衡的最少丢弃次数
    def minArrivalsToDiscard(self, arrivals: list[int], w: int, m: int) -> int:
        n = len(arrivals)
        ans = 0
        cnt = defaultdict(int)
        for i, x in enumerate(arrivals):
            # x数量达到m时才选择丢弃可以保证丢弃次数最小
            if cnt[x] == m:
                ans += 1
                # 此时x已经被丢弃, 在x离开窗口时不能直接删除cnt[x], 将x标记为0, 在移除x（此时arrivals[i] == 0）时，执行cnt[0]--, 不影响最终结果
                arrivals[i] = 0
            else:
                cnt[x] += 1
            # 计算窗口左端点
            left = i - w + 1

            if left >= 0:
                cnt[arrivals[left]] -= 1

        return ans
    
    # LC.1052.爱生气的书店老板
    # 逆向思维: 记录minutes区间内老板生气的最大值 + 记录mintues区间外满意的顾客人数
    def maxSatisfied(self, customers: list[int], grumpy: list[int], minutes: int) -> int:
        s = [0, 0]
        maxNum = 0
        for i, x in enumerate(customers):
            # 记录minutes区间外满意的顾客人数
            if grumpy[i] == 0:
                s[0] += x
            else:
                s[1] += x
            
            left = i - minutes + 1

            if left < 0:
                continue

            maxNum = max(maxNum, s[1])

            if grumpy[left]:
                s[1] -= customers[left]
            
        # 老板不生气时满意的顾客人数 + 使用能力后满意的顾客人数
        return s[0] + maxNum
    
    # LC.3652.按策略买卖股票的最佳时机
    def maxProfit(self, prices: list[int], strategy: list[int], k: int) -> int:
        ans = 0

        return ans

    # 不定长滑动窗口
    # 求最长子数组, 求最短子数组, 求子数组的个数
    # 使用队列来表示不定长的滑动窗口

    # 求最长子数组 / 越短越合法
    # LC.3.无重复字符的最长子串
    def lengthOfLongestSubstring(self, s: str) -> int:
        queue = deque()
        cnt = defaultdict(int)
        ans = 0
        for i, x in enumerate(s):
            queue.append(x)
            cnt[x] += 1

            while cnt[x] > 1:
                y = queue.popleft()
                cnt[y] -= 1
            ans = max(ans, len(queue))
        return ans
    
    # LC.3090.每个字符最多出现两次的子字符串
    def maximumLengthSubstring(self, s: str) -> int:
        ans = 0
        cnt = defaultdict(int)
        queue = deque()
        for i, x in enumerate(s):
            cnt[x] += 1
            queue.append(x)

            while cnt[x] > 2:
                left_char = queue.popleft()
                cnt[left_char] -= 1
            
            ans = max(ans, len(queue))

        return ans
    
    # LC.1493.删掉一个元素以后全为1的子数组
    def longestSubarray(self, nums: list[int]) -> int:
        ans = 0
        left = 0
        cnt  = [0, 0]
        for i, x in enumerate(nums):
            cnt[x] += 1

            while cnt[0] > 1:
                cnt[nums[left]] -= 1
                left += 1
            
            ans = max(ans, i - left + 1)
                            
        return ans - 1
    
    # LC.3634.使数组平衡的最少移除数目
    def minRemoval(self, nums: list[int], k: int) -> int:
        # 题目不要求数组保持原顺序, 在排序后枚举数组的最大值
        nums.sort()
        max_save = left = 0
        for i, x in enumerate(nums):
            while nums[left] * k < x:
                left += 1
            # 区间长度
            max_save = max(max_save, i - left + 1)

        return len(nums) - max_save

    # LC.1208.尽可能使字符串相等
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        ans = left = 0
        count = 0
        n = len(s)
        for i in range(n):
            temp_a, temp_b = s[i], t[i]
            if temp_a != temp_b:
                count += abs(ord(temp_a) - ord(temp_b))
                while count > maxCost:
                    count -= abs(ord(s[left]) - ord(t[left]))
                    left += 1
            ans = max(ans, i - left + 1)
        return ans
    
    # LC.1695.删除子数组的最大得分
    def maximumUniqueSubarray(self, nums: list[int]) -> int:
        ans = left = 0
        sum_num = 0
        cnt = defaultdict(int)
        for i, x in enumerate(nums):
            cnt[x] += 1
            sum_num += x

            while cnt[x] > 1:
                y = nums[left]
                sum_num -= y
                cnt[y] -= 1
                left += 1

            ans = max(ans, sum_num)
        return ans
    
    # LC.2958.最多K个重复元素的子数组
    def maxSubarrayLength(self, nums: list[int], k: int) -> int:
        ans = left = 0
        length = 0
        cnt = defaultdict(int)
        for i, x in enumerate(nums):
            cnt[x] += 1
            length += 1

            while cnt[x] > k:
                y = nums[left]
                cnt[y] -= 1
                length -= 1
                left += 1
            ans = max(ans, length)
        return ans
    
    # LC.考试的最大困扰度
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        ans = left = 0
        cnt = defaultdict(int)
        for i, x in enumerate(answerKey):
            cnt[x] += 1

            while cnt['T'] > k and cnt['F'] > k:
                cnt[answerKey[left]] -= 1
                left += 1

            ans = max(ans, i - left + 1) 

        return ans
    
    # LC.1004.最大连续1的个数3
    def longestOnes(self, nums: list[int], k: int) -> int:
        ans = left = 0
        zero_nums = 0
        for i, x in enumerate(nums):
            if x == 0:
                zero_nums += 1

            while zero_nums > k:
                if nums[left] == 0:
                    zero_nums -= 1
                left += 1
            ans = max(ans, i - left + 1)

        return ans
    
    # LC.2730.找到最长的半重复子字符串
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        ans = 1 
        left = cnt = 0
        for right in range(1, len(s)):
            if s[right] == s[right-1]:
                cnt += 1 
            
            while cnt > 1:

                if s[left] == s[left+1]:
                    cnt -= 1

                left += 1
            ans = max(ans, right - left + 1)
        return ans
    
    # LC.2779.数组的最大美丽值
    def maximumBeauty(self, nums: list[int], k: int) -> int:
        ans = left = 0
        nums.sort()
        for i, x in enumerate(nums):

            while x - nums[left] > 2*k:
                left += 1
            ans = max(ans, i - left + 1)

        return ans

    # LC.3258.统计满足K约束的子字符串数量1
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        ans = left = 0
        cnt = [0, 0]
        for i, x in enumerate(s):
            if x == '0':
                cnt[0] += 1
            else:
                cnt[1] += 1
            
            while cnt[0] > k and cnt[1] > k:
                if s[left] == '0':
                    cnt[0] -= 1
                else:
                    cnt[1] -= 1
                left += 1
            ans += i - left + 1
        return ans

    # LC.1358.包含所有三种字符的子字符串数目
    def numberOfSubstrings(self, s: str) -> int:
        ans = left = 0
        cnt = Counter()
        for i, x in enumerate(s):
            cnt[x] += 1

            while len(cnt) == 3:
                cnt[s[left]] -= 1
                if cnt[s[left]] == 0:
                    del cnt[s[left]]

                left += 1
                
            ans += left
        return ans

    # LC.930.和相同的二元子数组
    # 恰好型滑动窗口：计算有多少个元素和恰好等于k的子数组个数 => 恰好大于等于k的子数组个数减去恰好大于等于k+1的子数组个数
    def numSubarraysWithSum(self, nums: list[int], goal: int) -> int:
        def window(nums: list[int], goal: int) -> int:
            ans = left = sum_ = 0
            for i, x in enumerate(nums):
                sum_ += x
                while left <= i and sum_ > goal:
                    sum_ -= nums[left]
                    left += 1
                ans +=  i - left + 1
            return ans
        
        return window(nums, goal) - window(nums, goal - 1)
    
    # LC.1248.统计优美子数组
    def numberOfSubarrays(self, nums: list[int], k: int) -> int:
        def window(nums: list[int], k: int) -> int:
            left = ans = 0
            cnt = 0
            for i, x in enumerate(nums):
                if x % 2 == 1:
                    cnt += 1

                while cnt > k:
                    if nums[left] % 2 == 1:
                        cnt -= 1
                    left += 1 
                ans += i - left + 1 # 计算 奇数 <= k 的子数组数量
            return ans
        return window(nums, k) - window(nums, k-1)
    
    '''
        双指针
    '''
    # 相向双指针
    # LC.3643.垂直翻转子矩阵
    def reverseSubmatrix(self, grid: list[list[int]], x: int, y: int, k: int) -> list[list[int]]:
        ans = grid.copy()
        row_left, row_right = x, x + k - 1

        while row_left < row_right:
            for j in range(y, y+k):
                ans[row_left][j], ans[row_right][j] = ans[row_right][j], ans[row_left][j]
            row_left += 1
            row_right -= 1
        
        return ans
    
    # LC.345.反转字符串中的元音字母
    def reverseVowels(self, s: str) -> str:
        i, j = 0, len(s)-1
        slist = list(s)
        while i < j:

            if slist[i] not in 'aeiouAEIOU':
                i += 1

            if slist[j] not in 'aeiouAEIOU':
                j -= 1

            if slist[i] in 'aeiouAEIOU' and slist[j] in 'aeiouAEIOU':
                slist[i], slist[j] = slist[j], slist[i]
                i += 1
                j -= 1

        return ''.join(slist)
    
    # LC.1750.删除字符串两端相同的字符后的最短长度
    def minimumLength(self, s: str) -> int:
        i, j = 0, len(s)-1
        while i < j and s[i] == s[j]:
            temp = s[i]
            while i <= j and s[i] == temp:
                i += 1
            while i <= j and s[j] == temp:
                j -= 1
        return j - i + 1
    
    # LC.2105给植物浇水
    def minimumRefill(self, plants: list[int], capacityA: int, capacityB: int) -> int:
        tempa, tempb = capacityA, capacityB
        i, j = 0, len(plants)-1
        cnt = 0
        for _ in range(math.ceil(len(plants))):

            if i == j:
                if tempb > tempa:
                    if tempb < plants[i]:
                        tempb -= plants[i]
                        cnt += 1
                else:
                    if tempa < plants[i]:
                        tempa -= plants[i]
                        cnt += 1
                break



            if tempa >= plants[i]:
                tempa -= plants[i]
            else:
                cnt += 1
                tempa = capacityA - plants[i]

            if tempb >= plants[j]:
                tempb -= plants[j]
            else:
                cnt += 1
                tempb = capacityB - plants[j]

            i += 1
            j -= 1
        return cnt
    
    # LC.658.找到K个最接近的数
    def findClosestElements(self, arr: list[int], k: int, x: int) -> list[int]:
        left, right = 0, len(arr)-1
        while right - left + 1 < k:
            temp_left = abs(arr[left] - k)
            temp_right = abs(arr[right] - k)

            if temp_left < temp_right:
                right -= 1
            else:
                left += 1
            
        return arr[left:right+1]
    
    # LC.1471.数组中的K个最强值
    def getStrongest(self, arr: list[int], k: int) -> list[int]:
        arr.sort()
        m = arr[math.ceil(len(arr) / 2) - 1]
        left, right = 0, len(arr)-1
        ans = [0] * k
        for i in range(k):
            if abs(arr[left] - m) > abs(arr[right] - m):
                ans[i] = arr[left]
                left += 1
            else:
                ans[i] = arr[right]
                right -= 1
        return ans
    
    # LC.167.两数之和2
    def twoSum(self, numbers: list[int], target: int) -> list[int]:
        left, right = 0, len(numbers)-1
        ans = [0, 0]
        while left < right:
            sum_ = numbers[left] + numbers[right]
            if sum_ > target:
                right -= 1
            elif sum_ < target:
                left += 1
            else:
                ans = [left+1, right+1]
                break
        return ans
    
    # LC.633.平方数之和
    def judgeSquareSum(self, c: int) -> bool:
        left, right = 0, math.isqrt(c)

        while left <= right:
            temp = math.pow(left, 2) + math.pow(right, 2)
            if temp > c:
                right -= 1
            elif temp < c:
                left += 1
            else:
                return True
        
        return False
    
    # LC.2824.统计和小于目标的下标对数目
    def countPairs(self, nums: list[int], target: int) -> int:
        nums.sort()
        ans = 0
        left, right = 0, len(nums)-1
        while left < right:
            sum_ = nums[left] + nums[right]
            if sum_ >= target:
                right -= 1
            else:
                ans += right - left
                left += 1
        return ans 
    
    # LC.2563.统计公平数对的数目
    def countFairPairs_0(self, nums: list[int], lower: int, upper: int) -> int:
        left, right = 0, len(nums)-1
        nums.sort()
        cnt = 0
        while left < right:
            sum_ = nums[left] + nums[right]
            if lower <= sum_ <= upper:
                temp_right = right
                # O(n^2)
                # 超时
                while left < temp_right and nums[left] + nums[temp_right] >= lower:
                    cnt += 1
                    temp_right -= 1
                left += 1
            elif sum_ > upper:
                right -= 1
            elif sum_ < lower:
                left +=1 
        return cnt

    # 两次双指针
    # 枚举i, 统计有多少合法的j
    # 即：nums[j] <= upper - nums[i]的数量 减去 nums[j] < lower - nums[i]的数量
    def countFairPairs_1(self, nums: list[int], lower: int, upper: int) -> int:
        nums.sort()
        def count(upper):
            res = 0
            j = len(nums)-1
            for i, x in enumerate(nums):
                while i < j and nums[j] > upper - x:
                    j -= 1
                if i == j:
                    break
                # j满足条件后, j左侧的组合也均满足
                res += j - i
            return res
        return count(upper) - count(lower-1)
    
    # LC.28.采购方案
    def purchasePlans(self, nums: list[int], target: int) -> int:
        nums.sort()
        left, right = 0, len(nums)-1
        ans = 0
        while left < right:
            sum_ = nums[left] + nums[right]
            if sum_ > target:
                right -= 1
            else:
                ans += right - left
                left += 1
        return int(math.fmod(ans, 1000000007))

    # LC.16.最接近的三数之和
    # 基于三数之和的做法
    def threeSumClosest(self, nums: list[int], target: int) -> int:
        nums.sort()
        minVal = float('inf')
        ans = 0
        for i in range(len(nums)-2):
            x = nums[i]
            left, right = i+1, len(nums)-1
            while left < right:
                if x + nums[left] + nums[right] == target:
                    return target

                if x + nums[left] + nums[right] < target:
                    if minVal > target - (x+nums[left]+nums[right]):
                        minVal = target - (x+nums[left]+nums[right])
                        ans = (x+nums[left]+nums[right])
                    left += 1
                
                if x + nums[left] + nums[right] > target:
                    if minVal > (x+nums[left]+nums[right]) - target:
                        minVal = (x+nums[left]+nums[right]) - target
                        ans = (x+nums[left]+nums[right])
                    right -= 1
        return ans
    
    # LC.18.四数之和
    # 整体思路和三数之和相同, 固定两个数字, 使用双指针枚举剩余的两个数字
    def fourSum(self, nums: list[int], target: int) -> list[list[int]]:
        nums.sort()
        ans = []
        for i in range(len(nums)-3):
            x = nums[i]
            if i and x == nums[i-1]:
                continue
            for j in range(len(nums)-2):
                y = nums[j]
                if j > i+1 and y == nums[j-1]:
                    continue
                l, r = j+1, len(nums)-1
                while l < r:
                    sum_ = x + y + nums[l] + nums[r]
                    if sum_ == target:
                        ans.append([x, y, nums[i], nums[j]])
                    elif sum_ > target:
                        r -= 1
                    else:
                        l += 1
        return ans
    
    # 同向双指针
    # LC.611.有效三角形的个数
    # 枚举最长边
    def triangleNumber(self, nums: list[int]) -> int:
        ans = 0
        nums.sort()
        for c in range(2, len(nums)):
            a, b = 0, c-1
            while a < b:
                if nums[a] + nums[b] > c:
                    ans += b - a
                    b -= 1
                else:
                    a += 1
        return ans
    
    # LC.1577.数的平方等于两数乘积的方法数
    def numTriplets(self, nums1: list[int], nums2: list[int]) -> int:
        def count_num(nums1, nums2) -> int:
            ans = 0
            nums1.sort()
            nums2.sort()
            cnt = Counter(nums2)
            for i, x in enumerate(nums1):
                j, k = 0, len(nums2)-1
                while j < k:
                    s = x ** 2 - nums2[j] * nums2[k]
                    if s > 0:
                        j += 1
                    elif s < 0:
                        k -= 1
                    else: # 存在 x ** 2 == nums2[j] * nums2[k]
                        if nums2[j] == nums2[k]: # j...k之间的元素全部相等
                            ans += (cnt[nums2[j]] * (cnt[nums2[j]]-1)) // 2
                        elif nums2[j] != nums2[k]:
                            ans += cnt[nums2[j]] * cnt[nums2[k]]
                        j += 1
                        # 重复元素已经被计算过了
                        while j < k and nums2[j] == nums2[j-1]:
                            j += 1
                        k -= 1
                        while k > j and nums2[k] == nums2[k+1]:
                            k -= 1
            return ans
        return count_num(nums1,nums2) + count_num(nums2, nums1)