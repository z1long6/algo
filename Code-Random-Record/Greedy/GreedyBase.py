class Solution:
    # LC.455.分发饼干
    # 贪心问题 对饼干和胃口数组排序后,使用尽可能小的饼干满足尽可能大的胃口
    def findContentChildren(self, g: list[int], s: list[int]) -> int:
        g, s = sorted(g), sorted(s)
        res = 0

        j = 0
        for i in range(len(s)):
            if j == len(g):
                break

            if s[i] >= g[j]:
                res += 1
                j += 1 
                continue

            if s[i] < g[j]:
                continue
        
        return res
    
    # LC.376.摆动序列
    def wiggleMaxLength(self, nums: list[int]) -> int:
        if len(nums) <= 1:
            return len(nums)

        prediff, curdiff = 0, 0
        # 默认数组最右侧有一个峰值
        res = 1
        for i in range(len(nums)-1):
            curdiff = nums[i+1] - nums[i]
            if (prediff <= 0 and curdiff > 0) or (prediff >= 0 and curdiff < 0):
                res += 1
                # 只有当发现峰值时才更新prediff，解决在一个单调坡中出现平坡的情况，即在一个单调坡中出现平坡，此平坡不能被视为峰值变化，因为此平坡仍然处于单调坡中
                prediff = curdiff
        
        return res
    
    # LC.53.最大子数组
    def maxSubArray(self, nums: list[int]) -> int:
        # 暴力解
        # temp_max = float('-inf')
        # for i in range(0, len(nums)):
        #     for j in range(0, len(nums)-i):
        #         temp_max = max(temp_max, sum(nums[j:j+i+1]))
        # return temp_max
        # 贪心策略
        
        result = float('-inf')
        count = 0
        for i in range(len(nums)):
            count += nums[i]
            if count > result:
                result = count

            if count < 0:
                count = 0
                continue
        return int(result)
    
    # LC.122.买卖股票的最佳时机Ⅱ
    # 买卖股票的收益可以被拆分为每两天之差的和
    def maxProfit(self, prices: list[int]) -> int:
        res = 0
        temp_profit = []
        for i in range(len(prices)-1):
            temp_profit.append(prices[i+1] - prices[i])

        for i in range(len(temp_profit)):
            if temp_profit[i] >= 0:
                res += temp_profit[i]
        return res

    # LC.55.跳跃游戏
    def canJump(self, nums: list[int]) -> bool:
        if len(nums) == 1:
            return True
        
        end_index = 0
        for i in range(len(nums)):
            if i > end_index:
                break

            end_index = max(end_index, nums[i]+i)
            if end_index >= len(nums)-1:
                return True
            
        return False
    
    # LC.45.跳跃游戏Ⅱ
    # 维护当前可到范围和下一步可到达范围
    def jump(self, nums: list[int]) -> int:
        if len(nums) == 1:
            return 0

        res = 0
        i = 0
        end_index = 0
        while i <= end_index:
            for i in range(i, end_index+1):
                end_index = max(end_index, nums[i]+i)
                if end_index >= len(nums) - 1:
                    return res+1
            res += 1

    # LC.1005.K次取反后最大化数组和
    def largestSumAfterKNegations(self, nums: list[int], k: int) -> int:
        nums.sort()
        flag = float('-inf')
        res = 0
        for i in range(len(nums)):
            # 找到第一个不为负的数字
            if nums[i] >= 0:
                flag = i
                break

        if flag < 0:
            flag = len(nums)
        
        if k >= flag:

            if (k - flag) % 2 == 0:
                # 为偶数
                res = sum([abs(x) for x in nums])
            else:
                # 为奇数
                temp = min([abs(x) for x in nums])
                res = sum([abs(x) for x in nums]) - temp * 2

            return res
        
        else:
            for i in range(0, k):
                nums[i] = -nums[i]
            return sum(nums)
    
    # LC.134.加油站
    def canCompleteCircuit(self, gas: list[int], cost: list[int]) -> int:
        

        # def judge(j: int) -> bool:
        #     index = j 
        #     rest = 0
        #     while rest >= 0:
        #         rest += (gas[index] - cost[index])

        #         if rest < 0:
        #             return False
                
        #         index +=1
        #         index = index % len(gas)

        #         if index == j:
        #             return True

        #     return False


        # # 暴力遍历
        # for i in range(len(gas)):
        #     # 起始位置是i
        #     # 判断是否满足条件
        #     if judge(i):
        #         return i
            
        # return -1

        # 贪心
        ans = min_s = s = 0
        for i, (g,c) in enumerate(zip(gas, cost)):
            s = s + g - c # 维护当前油量

            # 存在更小油量，更新出发点
            if s < min_s:
                min_s = s
                ans = i + 1 # 在执行s-c时，汽车移动到i+1
        
        return -1 if s < 0 else ans
    
    # LC.135.分发糖果
    def candy(self, ratings: list[int]) -> int:
        res = [1 for i in range(len(ratings))]
        # 右侧孩子ratings大于左侧孩子：从前向后遍历
        for i in range(1, len(ratings)):
            res[i] = res[i-1] + 1 if ratings[i] > ratings[i-1] else res[i]
        
        # 左侧孩子ratings大于右侧孩子：从后向前遍历
        for i in range(len(ratings)-2, -1, -1):
            res[i] =  max(res[i], res[i+1]+1) if ratings[i] > ratings[i+1] else res[i]

        return sum(res)
    
    # LC.860.柠檬水找零
    def lemonadeChange(self, bills: list[int]) -> bool:
        dict_ = {5: 0, 10: 0}

        for i in range(len(bills)):
            if bills[i] == 5:
                dict_[5] += 1
            elif bills[i] == 10:
                dict_[10] += 1
                if dict_[5] == 0:
                    return False
                else:
                    dict_[5] -= 1

            elif bills[i] == 20:
                if dict_[10] == 0:
                    if dict_[5] >= 3:
                        dict_[5] -= 3
                    else:
                        return False
                else:
                    dict_[10] -= 1
                    if dict_[5] >= 1:
                        dict_[5] -=1
                    else:
                        return False

        return True
    
    # LC.406.根据身高重建队列
    # 一个带排序列表中存在两个排序指标时，考虑使用其中一个作为主指标（一个指标选择正序排序，另一个指标选择倒序排序）
    # 在完成排序后再进行插入
    def reconstructQueue(self, people: list[list[int]]) -> list[list[int]]:
        people.sort(key = lambda x : (-x[0], x[1]))
        res = []
        for p in people:
            if len(res) <= p[1]:
                res.append(p)
            else:
                res.insert(p[1], p)
        return res
    
    # LC.452.用最小数量的箭引爆气球
    def findMinArrowShots(self, points: list[list[int]]) -> int:
        # res = 1

        # if len(points) == 1:
        #     return res
        
        # points.sort(key = lambda x : x[0])
        # cur_min_right = points[0][1]

        # for point in points:

        #     if point[0] > cur_min_right:
        #         res += 1 
        #         cur_min_right = point[1]
        #     else:
        #         cur_min_right = min(cur_min_right, point[1])
        # return res
        res = 0
        points.sort(key=lambda x : x[1])
        min_right = float('-inf')
        for left, right in points:
            if left > min_right:
                res += 1
                min_right = right
        return right

        # i = 0
        # j = i + 1

        # while j <= len(points) - 1:
        #     if points[j][0] <= points[i][1]:
        #         # 重叠区间
        #         j += 1
        #         continue
        #     else:
        #         # 出现不重叠区间
        #         res += 1
        #         i = j
        #         j = i + 1
        return res
        
    # LC.435.无重叠区间
    def eraseOverlapIntervals(self, intervals: list[list[int]]) -> int:
        intervals.sort(key=lambda x : x[1])
        min_right = float('-inf')
        cnt = 0
        for left, right in intervals:
            # 更新不重叠的子区间
            if left >= min_right:
                cnt += 1
                min_right = right
        return len(intervals) - cnt
    
    # LC.763.划分字母区间
    def partitionLabels(self, s: str) -> list[int]:
        # 暴力遍历
        # res = []
        # for char_ in s:
        #     res.append(char_)

        #     i = 0
        #     while i < len(res) - 1:
        #     # for i in range(len(res)-1):
        #         # 检查是否在res中出现
        #         if char_ in res[i]: 
        #             # 从i开始将后续所有字符串合并成同一字符串
        #             temp_str = ''.join(res[i:])
        #             # 替换原列表
        #             res[i:] = [temp_str]
        #         i += 1      

        # return [len(x) for x in res] 

        # 贪心
        cnt_dict = {}
        for idx, str_ in enumerate(s):
            cnt_dict[str_] = idx
        start = 0
        end = 0
        res = []
        for idx, str_ in enumerate(s):
            end = max(end, cnt_dict[str_])
            if idx == end:
                res.append(end - start + 1)
                start = idx + 1
        return res
    
    # LC.56.合并区间
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        if len(intervals) == 0:
            return intervals

        intervals.sort(lambda x : x[0])
        result = []
        result.append(intervals[0])

        for i in range(1, len(intervals)):

            # 判断区间
            if result[-1][1] >= intervals[i][0]:
                # merge
                result[-1][1] = max(intervals[i][1], result[-1][1])
            else:
                # next one
                result.append(intervals[i])
        
        return result
    
    # def judge(n: int) -> bool:
    #     str_ = str(n)
    #     for i in range(len(str_) - 1):
    #         if str_[i] <= str_[i+1]:
    #             continue
    #         else:
    #             return False
    #     return True
    
    # # LC.738.单调递增的数字(暴力)
    # def monotoneIncreasingDigits(self, n: int) -> int:
        
    #     while n > 0:
    #         if self.judge(n):
    #             return n
    #         else:
    #             n -= 1

    #     return n
    # 贪心
    def monotoneIncreasingDigits(self, n: int) -> int:
        str_ = str(n)

        for i in range(len(n) - 1, 0,  -1):
            if str_[i-1] > str_[i]:
                str_[i-1] = str(int(str_[i-1]) - 1)

                for j in range(i, len(n)):
                    str_[j] = '9'
        return int(''.join(str_))
        