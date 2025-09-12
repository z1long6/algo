from collections import Counter
class Solution:
    
    # LC.509.斐波那契数列
    '''
        dp[i]代表第i个数字
    '''
    def fib(self, n: int) -> int:
        result = []
        for i in range(0, n+1):
            # 递推
            if i >= 2:
                result.append(result[i-1] + result[i-2])
            elif i == 0:
                result.append(0)
            elif i == 1:
                result.append(1)

        return result[n]
    
    # LC.70.爬楼梯
    '''
        dp[i]代表到达第i阶楼梯的方法数量
    '''
    def climbStairs(self, n: int) -> int:

        dp = [0, 1, 2]
        for i in range(3, n+1):
            # 递推
            dp.append(dp[i-1] + dp[i-2])
        return dp[n]
    
    
    # LC.746.使用最小花费爬楼梯
    '''
        反向遍历DP数组
        dp[i]代表第i阶楼梯到达楼顶的最小花费
    '''
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        dp = [0] * len(cost)
        dp[-1] = cost[-1]
        dp[-2] = cost[-2]
        for i in range(len(cost)-3, 0, -1):
            dp[i] = cost[i] + min(dp[i+1], dp[i+2])
        return min(dp[0], dp[1])
    
    # LC.62.不同路径
    '''
        dp[i][j]含义 从start到达dp[i][j]的方法数
        二维dp数组
        初始化dp数组第一行, 第一列为1
    '''
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0] * n for _ in range(m)]
        for j in range(1, n):
            dp[0][j] = 1

        for i in range(1, m):
            dp[i][0] = 1
        
        for i in range(1, m):
            for j in range(1, n):
                # 递推公式
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[-1][-1]
    
    def element_in_2d_list(self, arr, target):
        return any(target in row for row in arr)

    # LC.63.不同路径2
    def uniquePathsWithObstacles(self, obstacleGrid: list[list[int]]) -> int:
        if obstacleGrid[0][0] == 1 or obstacleGrid[-1][-1] == 1:
            return 0

        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        dp = [[0] * n for _ in range(m)]

        # init
        for i in range(0, m):
            if obstacleGrid[i][0] == 1:
                for k in range(i, m):
                    dp[k][0] = 0
                break
            else:
                dp[i][0] = 1
        
        for j in range(0, n):
            if obstacleGrid[0][j] == 1:
                for k in range(j, n):
                    dp[0][k] = 0
                break
            else:
                dp[0][j] = 1

        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
                else:
                    dp[i][j] = 0
        
        return dp[-1][-1]
    
    # LC.343.整数拆分
    '''
        dp[i]: 整数i被拆分后最大积
        状态转移方程 dp[i]= max(dp[i], j*(i-j), j*dp[i-j]),其中,j*(i-j)是将i拆分为两个数字相乘,j*dp[i-j]是将数字拆解为多个数字相乘
    '''
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n+1)
        # init
        dp[2] = 2
        for i in range(3, n):
            for j in range(1, i):
                # 递推
                dp[i]= max(dp[i], j*(i-j), j*dp[i-j])

        return dp[n]
        
    # LC.96.不同的二叉搜索树
    '''
        dp[i] 给定整数i,形成二叉搜索树的种类数量
        状态转移方程 dp[i] = dp[j] * dp[i-j-1] 对于整数i, 形成的二叉搜索树分别以1,2,3 ... i-1, i作为根节点, 则i的左子树有1, 2, 3, 4, ..., i-1 个根节点, i的右子树的节点个数为 i-j-1 
    '''
    def numTrees(self, n: int) -> int:
        dp = [0] * (n+1)
        dp[0], dp[1] = 1, 1
        for i in range(2, n+1):
            j = 0
            while j < i:
                dp[i] += dp[j] * dp[i-j-1]
                j += 1
        return dp[n]
    
    # 携带研究材料
    '''
        二维dp数组
        dp[i][j] [0, i]物品任选, 放入容量为j的背包中的最大价值
        状态转移方程 dp[i][j] = max(dp[i-1][j], dp[i-1][j-weight[i]] + value[i])
    '''
    def bags_01(self, m: int, n: int, weight: list, value: list):
        dp = [[0] * (n+1) for _ in range(m)]
        # init
        for j in range(weight[0], n+1):
                dp[0][j] = value[0]
        
        # dp
        for i in range(1, m):
            for j in range(1, n+1):
                if j < weight[i]: # 背包空间不足以放下物品i
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j - weight[i]] + value[i])
        
        return dp[-1][-1]

    '''
        一维dp数组: 01背包问题在先遍历物品, 根据状态转移方程, 后遍历背包的情况下, dp数组的下一个状态仅由前一行来决定, 可以使用滚动数组
        dp[j] 背包容量为j所具有的最大价值
        状态转移方程 dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    '''
    def bags_01_one_dim(self, m: int, n: int, weight: list, value: list):
        dp = [0] * (n+1)

        # dp
        for i in range(0, m):
            for j in range(n, -1, -1):
                if j >= weight[i]:
                    dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

        return dp[-1]
    
    # LC.416.分割等和子集
    ''' 
        在本题中, 物品的价值和重量相同, 均为nums数组
        dp[i][j] 背包重量为j时, 任选[0...i]的物品, 最大价值为dp[i][j]
        dp[i][j] = max(dp[i-1][j], dp[i-1][j - nums[i]] + nums[i])
    '''
    def canPartition(self, nums: list[int]) -> bool:
        # 奇数
        if sum(nums) % 2 == 1:
            return False
        # 物品个数
        len_ = len(nums)
        # 背包容量
        sum_ = sum(nums) // 2

        dp = [[0] * (sum_+1) for _ in range(len_)]
        # init
        for i in range(nums[0], sum_+1):
            dp[0][i] = nums[0]
        
        # dp
        for i in range(1, len_):
            for j in range(1, sum_+1):
                if j >= nums[i]:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j - nums[i]] + nums[i])
                else:
                    dp[i][j] = dp[i-1][j]
        
        if dp[-1][-1] == (sum(nums) // 2):
            return True
        else:
            return False
        
    # LC.1049.最后一块石头的重量2
    '''
        石块两两相减等价于将石块划分到两个不同group中相减, 求最小差值即求两个group相减的最小差值, 若每个group的和达到最大值且小于等于 (m / 2),
        01背包问题
    '''
    def lastStoneWeightII(self, stones: list[int]) -> int:
        len_ = len(stones)
        sum_ = sum(stones) // 2
        
        dp = [[0] * (sum_+1) for _ in range(len_)]

        # init
        for i in range(stones[0], sum_+1):
            dp[0][i] = stones[0]

        # dp
        for i in range(1, len_):
            for j in range(1, sum_+1):
                if j >= stones[i]:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j - stones[i]] + stones[i])
                else:
                    dp[i][j] = dp[i-1][j]
    
        return sum(stones) - dp[-1][-1] - dp[-1][-1] 

    # LC.494.目标和
    '''
        转化成01背包
        记正数为P 负数为N, 有: P + abs(N) = sum(nums) P + N = target => 2 * abs(N) = sum(nums) - target => abs(N) = (sum(nums) - target) / 2
        故本题转化为01背包为 使用nums数组装满容量为abs(N)的所有可能性 ps: 与一般01背包中装满容量为j的背包的物品最大价值不同
        dp[i][j]定义 任选[0...i]中的物品, 装满背包容量为j的所有情况
        状态转移方程 dp[i][j] = dp[i-1][j] or dp[i-1][j - nums[i]] + dp[i-1][j]
    '''
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        len_ = len(nums)
        sum_ = sum(nums)

        bags_weight = (sum_-target) // 2
        # 无解的情况
        if (sum_ - target) % 2 == 1 or abs(target) > sum_:
            return 0

        dp = [[0] * (bags_weight + 1) for _ in range(len_)]

        # init
        # 第一行初始化
        if nums[0] <= bags_weight:
            dp[0][nums[0]] = 1

        # 第一列初始化
        dp[0][0] = 1

        zero_nums = 0
        for i, x in enumerate(nums):
            if x == 0:
                zero_nums += 1
            dp[i][0] = pow(2, zero_nums)

        # dp
        for i in range(1, len_):
            for j in range(1, bags_weight+1):
                if j >= nums[i]:
                    dp[i][j] = dp[i-1][j] + dp[i-1][j - nums[i]]
                else:
                    dp[i][j] = dp[i-1][j]
        
        return dp[-1][-1]
    
    # 使用回溯解决本题
    def findTargetSumWays_backtracking(self, nums: list[int], target: int) -> int:
        len_ = len(nums)
        sum_ = sum(nums)
        if (sum_ - target) % 2 == 1 or abs(target) > sum_:
            return 0

        target_ = (sum_ - target) // 2

        result, path = [], []

        def backtracking(startindex: int) -> None:
            if sum(path) > target_:
                return 
            
            if sum(path) == target_:
                result.append(path.copy())

            if len(path) == len_:
                return
            
            for i in range(startindex, len_):
                path.append(nums[i])
                backtracking(i+1)
                path.pop()
        
        backtracking(0)

        return len(result)

    # LC.471.一和零
    '''
        解法1: 按照回溯算法，收集所有满足条件的节点，并获取最长节点
        解法2: dp
    '''

    def findMaxForm_1_judge(self, m:int, n:int, path: list[str]) -> bool:
        temp_str = ''.join(path)
        c = Counter(temp_str)
        if c['0'] <= m and c['1'] <= n:
            return True
        return False
    
    # 回溯
    def findMaxForm_1(self, strs: list[str], m: int, n: int) -> int:
        result, path = [], []

        def backtracing(startindex: int) -> None:

            # 收集子集节点
            # 判断path中存放的字符串是否符合条件
            if self.findMaxForm_1_judge(m, n, path):
                result.append(path.copy())
            
            if len (strs) == startindex:
                return 
            
            for i in range(startindex, len(strs)):
                path.append(strs[i])
                backtracing(i+1)
                path.pop()
        
        backtracing(0)
        max_len = 0
        for i, x in enumerate(result):
            max_len = max(max_len, len(x))
        return max_len

    # dp
    '''
        物品存在两个维度的价值，就使用两个维度来记录物品价值
        dp[i][j][k] 任选[0...i]个物品, 0的个数不超过j, 1的个数不超过k的最长子集
        状态转移方程
    '''
    def findMaxForm_2(self, strs: list[str], m: int, n: int) -> int:

        dp = [[[0] * (n+1) for _ in range(m+1)] for _ in range(len(strs) + 1)]

        for i, str_ in enumerate(strs):
            cnt0 = str_.count('0')
            cnt1 = len(str_) - cnt0
            for j in range(m+1):
                for k in range(n+1):
                    if j >= cnt0 and k >= cnt1:
                        dp[i+1][j][k] = max(dp[i][j][k], dp[i][j-cnt0][k-cnt1] + 1)
                    else:
                        dp[i+1][j][k] = dp[i][j][k]

        return dp[-1][-1][-1]
    
    # 完全背包标准问题
    def bags(self, weights, values, n, v):
        dp = [[0] * (v+1) for _ in range(n)]
        # init
        for j in range(weights[0], v+1):
            dp[0][j] = dp[0][j-weights[0]] + values[0]
        
        # dp
        for i in range(1, n):
            for j in range(v+1):
                if j >= weights[i]:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-weights[i]]+values[i])
                else:
                    dp[i][j] = dp[i-1][j]
        
        return dp[-1][-1]
    
    # LC.518.零钱兑换2
    '''
        物品价值数组和重量数组是同一个即coins, 最大重量为5
        dp[i][j]选取下标为[0...j]的物品, 装满容量为j的背包的方法有多少种, (ps: 这里dp数组的值是恰好装满背包容量为j时的方法数, 而非最大价值, 因此递推公式和求最大价值时不一样)
        递推公式两种情况: 
            j < coins[i] 不选物品i装满容量j的背包 dp[i][j] = dp[i-1][j]
            j >= coins[i] 可以再次选取物品i装满容量为j的背包 dp[i][j] = dp[i-1[j] + dp[i][j-coins[i]] 其中, 一个物品可以选取多次
    '''
    def change(self, amount: int, coins: list[int]) -> int:
        n = len(coins)
        dp = [[0] * (amount + 1) for _ in range(n)]

        # init
        for i in range(n):
            dp[i][0] = 1

        for j in range(0, amount+1):
            if (j % coins[0]) == 0:
                dp[0][j] = 1

        # dp
        for i in range(1, n):
            for j in range(amount+1):
                if j >= coins[i]:
                    dp[i][j] = dp[i-1][j] + dp[i][j-coins[i]]
                else:
                    dp[i][j] = dp[i-1][j]
        
        return dp[-1][-1]
    
    # 组合总和4
    '''
        排列问题使用一维dp
        本题同零钱兑换的不同点在于选取元素顺序不同产生的组合方式不同即排列
        dp[j] 装满容量为j的背包的所有排列数量
        状态转移方程: 
            j < nums[i] 不装物品i: dp[j] = dp[j]
            j >= nums[i] 装入物品i: dp[j] = dp[j] + dp[j - nums[i]]

    '''
    def combinationSum4(self, nums: list[int], target: int) -> int:
        dp = [0] * (target+1)
        dp[0] = 1

        for j in range(target+1):
            for i, x in enumerate(nums):
                if j >= x:
                    dp[j] = dp[j] + dp[j - x]
        
        return dp[-1]

    # 爬楼梯进阶版
    # 假设你正在爬楼梯, 需要n阶你才能到达楼顶, 每次你可以爬至多m(1 <= m < n)个台阶, 你有多少种不同的方法可以爬到楼顶呢?
    '''
        本题中, 每次可以选取[1...m]个台阶, 最终装满容量为j的背包的方法
        dp[j] 代表到达第j阶台阶的方法数量
        dp[j] = dp[j] + dp[j - i]
    '''
    def climbStairs2(self, n:int, m:int) -> int:
        dp = [0] * (n+1)
        dp[0] = 1
        for j in range(1, n+1):
            for i in range(1, m+1):
                if j >= i:
                    dp[j] = dp[j] + dp[j - i]
        return dp[n]
    
    # LC.322.零钱兑换
    '''
        dp[i][j] 从下标为[0...i]的硬币中选取, 恰好凑满和为amount时使用的最少硬币数量
        状态转移方程:
            dp[i][j] = min(dp[i-1][j], dp[i][j-coins[i]]+1)
    '''
    def coinChange(self, coins: list[int], amount: int):
        n = len(coins)
        dp = [[amount+1] * (amount+1) for _ in range(n)]
        # init
        for i in range(n):
            dp[i][0] = 0
        for j in range(amount+1):
            if j % coins[0] == 0:
                dp[0][j] = j // coins[0]
        # dp
        for i in range(1, n):
            for j in range(1, amount+1):
                if j >= coins[i]:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-coins[i]]+1)
                else:
                    dp[i][j] = dp[i-1][j]
        ans = dp[n][amount]
        return ans if ans < amount+1 else -1

    # LC.279.完全平方数
    '''
        dp[i][j] 选取下标为[0...i]的完全平方数, 凑满和为n的使用的平方数的最少数量
        状态转移方程:
            dp[i][j] = min(dp[i-1][j], dp[i][j-nums[i]]+1)
    '''
    def numSquares(self, n: int) -> int:
        nums = []
        # 创造物品数组
        for i in range(101):
            nums.append(i*i)
        
        dp = [[float('inf')] * (n+1) for _ in range(len(nums))]

        # init
        for i in range(len(nums)):
            dp[i][0] = 0

        for j in range(1, n+1):
            dp[0][j] = j
        
        # dp
        for i in range(1, len(nums)):
            for j in range(1, n+1):
                if j >= nums[i]:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-nums[i]]+1)
                else:
                    dp[i][j] = dp[i-1][j]
        ans = int(dp[-1][len(nums)])
        return ans
    
    # LC.139.单词拆分
    '''
        dp[i] == true 表示长度为i的子串能够被wordDict拼凑
        状态转移方程:
            if dp[j] and s[j:i] in wordDict:
                dp[i] == True
    '''
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        # 时间优化
        max_len = max(map(len, wordDict)) # 求出wordDict中的最大长度

        Set = set(wordDict)
        len_s = len(s)
        dp = [False] * (len_s+1)
        dp[0] = True
        for i in range(1, len_s+1):
            for j in range(max(0, i - max_len), i): # j只需要枚举max(0, i-max_len) 到 (i-1), 因为在wordDict中最长字符串长度为max_len => len(s[j:i]) <= max_len
                if dp[j] and s[j:i] in Set:
                    dp[i] = True
        return dp[len_s]