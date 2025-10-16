from collections import Counter, deque, defaultdict
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