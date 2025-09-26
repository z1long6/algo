from collections import Counter
from typing import Optional
from BinaryTree.MyTreeNode import TreeNode
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

    # LC.198.打家劫舍
    '''
        当前房屋选择与否与前一个和前两个房屋选择情况关联, 使用动态规划
        dp[i]在下标为[0, i]的房屋中选择偷取, 所能偷取到的最大价值是dp[i]
        状态转移方程:
            1. 选择当前房屋 dp[i] = dp[i-2] + nums[i]
            2. 不选择当前房屋 dp[i]= dp[i-1](也可能不选择前一个房屋)
            3. dp[i] = max(dp[i-2]+nums[i], dp[i-1])
    '''
    def rob(self, nums: list[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0], dp[1] = nums[0], max(nums[0], nums[1])
        for i, x in enumerate(nums[2:], 2):
            dp[i] = max(dp[i-2]+x, dp[i-1])

        return dp[n-1]
    
    # LC.213.打家劫舍2
    def rob2(self, nums: list[int]) -> int:
        n = len(nums)

        if n == 1:
            return nums[0]
        
        temp_nums1, temp_nums2 = nums.copy(), nums.copy()
        temp_nums1.pop()
        temp_nums2.reverse()
        temp_nums2.pop()
        return max(self.rob(temp_nums1), self.rob(temp_nums2))
    
    # LC.337.打家劫舍3
    '''
        树形结构中
        root = max(root.left+root.right, root-1)

    '''
    # 方法1 暴力递归
    def rob3_1(self, root: TreeNode) -> int:
        # 方法1 暴力
        def dfs(root: TreeNode):
            if root is None:
                return 0
            
            if root.left is None and root.right is None:
                return root.val
            
            # 选择当前root节点
            value1 = root.val

            if root.left is not None:
                value1 = value1 + dfs(root.left.left) + dfs(root.left.right)

            if root.right is not None:
                value1 = value1 + dfs(root.right.left) + dfs(root.right.right)
            
            # 不选择当前节点
            value2 = 0
            if root.left is not None:
                value2 += dfs(root.left) 
            if root.right is not None:
                value2 += dfs(root.right)
                
            return max(value1, value2)


        return dfs(root)
    '''
        1. 树形dp, 集合递归和动态规划
        2. dp 长度为2的数组, dp[0]代表不选择当前节点的偷窃总和最大值, dp[1]代表偷窃当前节点房子的偷窃总和最大值
        3. 状态转移方程:
                选择偷窃当前节点 var1 = root.val + left.dp[0] + right.dp[0], 选择偷窃当前节点, 则左孩子和右孩子不能被偷窃
                不选择偷窃当前节点, 则考虑偷窃子节点(但也可以选择不偷窃子节点) var2 = max(left.dp[0], left.dp[1]) + max(right.dp[0], right.dp[1])
        4. 必须使用后序遍历二叉树, 因为需要使用从下至上计算的递归结果, 父节点的值需要通过计算孩子节点的值来得到
    '''
    def rob3_2(self, root: Optional[TreeNode]) -> int:

        def rob3_dfs(root: Optional[TreeNode]) -> list[int]:
            if root is None:
                return [0, 0]
            
            var1, var2 = root.val, 0
            if root.left is not None:
                left_dp = rob3_dfs(root.left)
                var1 += left_dp[0]
                var2 = max(left_dp[0], left_dp[1])

            if root.right is not None:
                right_dp = rob3_dfs(root.right)
                var1 += right_dp[0]
                var2 += max(right_dp[0], right_dp[1])

            return [var2, var1]

        return max(rob3_dfs(root))
    
    # LC.121.买卖股票的最佳时机
    '''
        解法1: 暴力枚举; 解法2: 贪心, 枚举左侧最小值, 选择右侧最大值
        解法3: 动态规划
        本题只能交易一次, 但可以选择不交易
        1. dp数组定义
            dp[i][0]在第i天持有股票时, 手中的金钱
            dp[i][1]在第i天不持有股票, 手中的金钱
        2. 状态转移方程
            dp[i][0]
                前一天也持有股票 dp[i][0] = dp[i-1][0]
                前一天不持有股票, 第i天买入股票 dp[i][0] = -prices[i]
            dp[i][1]
                前一天不持有股票, 第i天不持有股票 dp[i][1] = dp[i-1][1]
                前一天持有股票, 第i天卖出股票 dp[i][1] = prices[i] + dp[i-1][0]
            dp[i][0] = max(dp[i-1][0], -prices[i])
            dp[i][1] = max(dp[i-1][1], prices[i] + dp[i-1][0])
        3. 初始化
            dp[0][0] = -prices[0]
            dp[0][1] = 0
        4. 遍历顺序
            按照天数
    '''
    def maxProfit(self, prices: list[int]) -> int:
        n = len(prices)
        if n == 1:
            return 0
        
        dp = [[0] * 2 for _ in range(n)]
        # init
        dp[0][0], dp[0][1] = -prices[0], 0
        # dp
        for i, x in enumerate(prices[1:], 1):
            dp[i][0] = max(dp[i-1][0], -x)
            dp[i][1] = max(dp[i-1][1], x + dp[i-1][0])

        return dp[n-1][1]
    
    # LC.122.买卖股票的最佳时机2
    '''
        1. 本题是每天都可以进行交易
        2. dp数组定义
            dp[i][0]: 第i天持有股票的最大收益
            dp[i][1]: 第i天不持有股票的最大收益
        3. 状态转移方程
            dp[i][0], 第i天持有股票, 分别是第i天卖出之前买的股票再重新买入股票 or 第i天继续持有前一天的股票 or 前一天不持有股票第i天买入股票
                dp[i][0] = dp[i-1][0] - prices[i] or dp[i-1][0]
            dp[i][1], 第i天不持有股票, 分别是第i天卖出股票 or 第i天仍然不持有股票
                dp[i][1] = dp[i-1][1] or dp[i-1][0] + prices[i]
    '''
    def maxProfit2(self, prices: list[int]) -> int:
        n = len(prices)
        if n == 1:
            return 0
        
        dp = [[0] * 2 for _ in range(n)]
        # init
        dp[0][0], dp[0][1] = -prices[0], 0
        # dp
        for i, x in enumerate(prices[1:], 1):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - x)
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + x)

        return dp[n-1][1]

    # LC.123.买卖股票的最佳时机3
    '''
        1. 本题限定最多可以完成两次交易(限定交易次数为j)
        2. 定义dp数组
            dp[i][0] 第i天不进行交易
            dp[i][1] 第i天在第1次持有股票所得现金
            dp[i][2] 第i天在第1次不持有股票所得现金
            dp[i][3] 第i天在第2次持有股票所得现金
            dp[i][4] 第i天在第2次不持有股票所得现金
        3. 状态转移方程
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
            dp[i][2] = max(dp[i-1][2], dp[i-1][1] + prices[i])
            dp[i][3] = max(dp[i-1][3], dp[i-1][2] - prices[i])
            dp[i][4] = max(dp[i-1][4], dp[i-1][3] + prices[i])
        4. 初始化
            dp[0][0] = 0 dp[0][1] = -prices[i] dp[0][2] = 0 dp[0][3] = -prices[i] dp[0][4] = 0
        5. 遍历顺序
            从前至后
    '''
    def maxProfit3(self, prices: list[int]) -> int:
        n = len(prices)
        dp = [[0] * 5 for _ in range(n)]
        # init 
        dp[0][0], dp[0][1], dp[0][2], dp[0][3], dp[0][4] = \
            0, -prices[0], 0, -prices[0], 0
        # dp
        for i, x in enumerate(prices[1:], 1):
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
            dp[i][2] = max(dp[i-1][2], dp[i-1][1] + prices[i])
            dp[i][3] = max(dp[i-1][3], dp[i-1][2] - prices[i])
            dp[i][4] = max(dp[i-1][4], dp[i-1][3] + prices[i])
        
        return dp[n-1][4]
    
    # LC.188.交易股票的最佳时机4
    '''
        本题同上, 但交易次数为k
    '''
    def maxProfit4(self, k:int, prices: list[int]) -> int:
        n = len(prices)            
        dp = [[0] * (2*k+1) for _ in range(n)]
        # init
        for j in range(2*k+1):
            if j % 2 == 1:
                dp[0][j] = -prices[0] # 当j为奇数, 表明第k次交易中持有股票, 故dp[0][j] = -prices[0]
        
        # dp
        for i, x in enumerate(prices[1:], 1):
            for j in range(0, 2*k-1, 2):
                # j为奇数, 持有股票, dp[i][j] = dp[i-1][j-1] - prices[i]
                dp[i][j+1] = max(dp[i-1][j+1], dp[i-1][j] - prices[i])
                dp[i][j+2] = max(dp[i-1][j+2], dp[i-1][j+1] + prices[i])

        return dp[n-1][2*k] 
    
    # LC.309.买卖股票的最佳时机含交易冷冻期
    '''
        1. 本题可以完成多次交易, 但如果在第i天卖出股票, 则第i+1天无法买入股票, 第i+1天为冷冻期
        2. 定义四个状态 0 买入股票状态 1 保持卖出股票状态 2 今天卖出股票 3 冷冻期状态
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i], dp[i-1][3] - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][3])
            dp[i][2] = dp[i-1][0] + prices[i]
            dp[i][3] = dp[i-1][2] 
    '''
    def maxProfit5(self, prices: list[int]) -> int:
        n = len(prices)
        dp = [[0] * 4 for _ in range(n)]
        # init
        dp[0][0], dp[0][1], dp[0][2], dp[0][3] = -prices[0], 0, 0, 0
        # dp
        for i, x in enumerate(prices[1:], 1):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - x, dp[i-1][3] - x)
            dp[i][1] = max(dp[i-1][1], dp[i-1][3])
            dp[i][2] = dp[i-1][0] + x
            dp[i][3] = dp[i-1][2]


        ans = 0
        for j in range(3):
            ans = max(ans, dp[n-1][j])
        return ans
    
    # LC.714.买卖股票的最佳时机含手续费
    '''
        本题同2, 在每次卖出股票获得收益时减去手续费
    '''
    def maxProfit6(self, prices: list[int], fee: int) -> int:
        n = len(prices)
        if n == 1:
            return 0
        
        dp = [[0] * 2 for _ in range(n)]
        # init
        dp[0][0], dp[0][1] = -prices[0], 0
        # dp
        for i, x in enumerate(prices[1:], 1):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - x)
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + x - fee)

        return dp[n-1][1]
    
    # LC.300.最长递增子序列
    '''
        1. dp数组定义
            dp[i] 以nums[i]结尾的子序列的最大长度
        2. 状态转移方程
            if dp[i] > dp[j]: dp[i] = max(dp[j] + 1, dp[i])
        3. 初始化
            dp[i] = 1
        4. 遍历顺序
            枚举下标j为[0...i-1]的数字, 比较dp[j]与dp[i]
    '''
    def lengthOfLIS(self, nums: list[int]) -> int:
        n = len(nums)
        dp = [1] * n
        result = 1
        for i, x in enumerate(nums[1:], 1):
            for j, y in enumerate(nums[:i]):
                if x > y:
                    dp[i] = max(dp[j]+1, dp[i])
            result = max(result, dp[i])
        return result
    
    # LC.674.最长连续递增子序列
    '''
        本题要求连续
    '''
    # 法1: 暴力
    def findLengthOfLCIS_0(self, nums: list[int]) -> int:
        result = 1
        temp = 1
        for i in range(len(nums)-1):
            if nums[i+1] > nums[i]:
                temp += 1
            else:
                temp = 1
            result = max(result, temp)
        return result

    # 法2: dp
    def findLengthOfLCIS_1(self, nums: list[int]) -> int:
        n = len(nums)
        dp = [1] * n
        result = 1
        for i, x in enumerate(nums[1:], 1):
            if x > nums[i-1]:
                dp[i] = dp[i-1] + 1
            else:
                dp[i] = 1
            result = max(result, dp[i])
        return result
    
    # LC.718.最长重复子数组
    '''
        子数组元素必须是连续的
        1. dp数组定义
            dp[i+1][j+1] 以nums1[i]结尾和以nums2[j]结尾的公共子数组的长度
        2. 状态转移方程
            if nums[i] == nums[j]: dp[i+1][j+1] = dp[i][j] + 1
        3. 初始化
            dp[i][0] = dp[0][j] = 0
    '''
    def findLength(self, nums1: list[int], nums2: list[int]) -> int:
        n, m = len(nums1), len(nums2)
        dp = [[0] * (n+1) for _ in range(m+1)]

        # dp
        for i, x in enumerate(nums1):
            for j, y in enumerate(nums2):
                if x == y:
                    dp[i+1][j+1] = dp[i][j] + 1
        return max(map(max, dp))
    
    # LC.1143.最长公共子序列
    '''
        子序列可以不连续
    '''
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n, m = len(text1), len(text2)
        dp = [[0] * (m+1) for _ in range(n+1)]

        # dp
        for i, x in enumerate(text1):
            for j, y in enumerate(text2):
                if x == y:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
        return max(map(max, dp))
    
    # LC.1035.不相交的线
    '''
        同最长公共子序列
    '''
    def maxUncrossedLines(self, nums1: list[int], nums2: list[int]) -> int:
        n, m = len(nums1), len(nums2)
        dp = [[0] * (m+1) for _ in range(n+1)]

        # dp
        for i, x in enumerate(nums1):
            for j, y in enumerate(nums2):
                if x == y:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
        return max(map(max, dp))
    
    # LC.53.最大子数组和
    # 法1: 贪心
    def maxSubArray_0(self, nums: list[int]) -> int:
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

    # dp
    '''
        1. dp数组定义
            dp[i] 以nums[i]结尾的子数组的和的最大值
        2. 状态转移方程
            dp[i] = max(dp[i-1]+nums[i], nums[i])
        3. 初始化
            dp[0] = max(0, nums[i])
        4. 遍历顺序
            从前至后
    '''
    def maxSubArray_1(self, nums: list[int]) -> int:
        m = len(nums)
        # init 
        dp = [0] * m
        dp[0] = max(0, nums[0])
        # dp
        for i, x in enumerate(nums[1:], 1):
            dp[i] = max(dp[i-1]+x, x)
        return max(dp)

    # LC.392.判断子序列
    # 遍历
    def isSubsequence_0(self, s: str, t: str) -> bool:
        flag = [False] * len(s)
        i, j = 0, 0
        while i < len(s):
            while j < len(t):
                if s[i] == t[j]:
                    flag[i] = True
                    j += 1
                    break
                j += 1 
            i += 1   
        return True if False not in flag else False
    # dp
    '''
        1. dp数组定义
            dp[i+1][j+1]: 以s[i], t[j]结尾的公共子序列的长度(相同子序列)
        2. 状态转移方程
            if s[i] == t[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = dp[i+1][j]
        3. 初始化
            dp[0][0] = 0
        4. 遍历顺序
    '''
    def isSubsequence_1(self, s: str, t: str) -> bool:
        m, n = len(s), len(t)
        # init
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i, x in enumerate(s):
            for j, y in enumerate(t):
                if x == y:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = dp[i+1][j]
        return True if dp[m][n] == m else False
    
    # LC.115.不同的子序列
    '''
        1. dp数组定义
            dp[i+1][j+1]: 以s[i]结尾的子序列中含有以t[j]结尾的子序列的个数
        2. 状态转移方程
            if s[i] == t[j]:
                dp[i+1][j+1] = dp[i][j] + dp[i][j+1]
            else:
                dp[i+1][j+1] = dp[i][j+1]
        3. 初始化
            dp[0][0] = 1
        4. 遍历顺序
            先遍历t, 再遍历s
    '''
    def numDistinct(self, s: str, t: str) -> int:
        # init
        dp = [[0] * (len(t)+1) for _ in range(len(s)+1)]
        for i in range(len(s)+1):
            dp[i][0] = 1
        # dp
        for i, x in enumerate(s):
            for j, y in enumerate(t):
                if x == y:
                    dp[i+1][j+1] = dp[i][j] + dp[i][j+1]
                else:
                    dp[i+1][j+1] = dp[i][j+1]

        return dp[len(s)][len(t)]
    
    # LC.583.两个字符串的删除操作
    '''
        本题转化为求s1, s2的最长公共子序列的长度为k, 则最小操作数量为 len(s1) + len(s2) - 2*k 
    '''
    def minDistance(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)
        dp = [[0] * (m+1) for _ in range(n+1)]

        # dp
        for i, x in enumerate(word1):
            for j, y in enumerate(word2):
                if x == y:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
        return n + m - 2 * max(map(max, dp))

    # LC.72.编辑距离
    '''
        1. dp数组定义
            dp[i][j]: 将word1[0...i-1]变成word2[0...j-1]所需要的最小操作数
        2. 状态转移方程
            if word1[i] == word2[j]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
            当word1[i] != word2[j]时, 需要进行删除, 添加, 置换操作, 其中删除和添加操作等价, 故考虑分别删除word1[i]或word2[j], 并取三者中的最小值
        3. 初始化
            dp[i][0] = i
            dp[0][j] = j
    '''
    def minDistance_2(self, word1: str, word2: str) -> int:
        # init
        dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
        for i in range(len(word1)+1):
            dp[i][0] = i
        for j in range(len(word2)+1):
            dp[0][j] = j
        # dp
        for i, x in enumerate(word1, 1):
            for j, y in enumerate(word2, 1):
                if x == y:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
        return dp[-1][-1]
    
    # LC.647.回文子串
    # 法1 暴力解法 时间复杂度O(n^3)
    def isPlindrome(self, s: str) -> bool:
        n = len(s) // 2
        for i in range(n):
            j = len(s) - i + 1
            if s[i] != s[j]:
                return False
        return True

    def countSubstrings_0(self, s: str) -> int:
        n = len(s)
        ans = 0
        for i in range(n):
            for k in range(i+1, n+1):
                temp_str = s[i:k]
                if self.isPlindrome(temp_str):
                    ans += 1
        return ans
    
    # dp
    '''
        1. dp数组定义
            dp[i][j]=True 代表下标为[i...j]的子串t为回文串
        2. 状态转移
            if s[i] == s[j]:
                if j - i <= 1: s[i]与s[j]相差位置不超过1位
                    ans += 1
                    dp[i][j] = True
                else: 相差位置超过1
                    if dp[i+1][j-1]: 判断剔除s[i], [s][j]的子串是否为回文串
                        ans += 1
                        dp[i][j] = True
            else:
                dp[i][j] = False
        3. 初始化
            dp[i][j] = False
        4. 遍历顺序 
            从下至上, 从左到右
    '''
    def countSubStrings_1(self, s: str) -> int:
        n = len(s)
        ans = 0
        dp = [[False] * n for _ in range(n)]

        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j]:
                    if j - i <= 1:
                        ans += 1
                        dp[i][j] = True
                    else:
                        if dp[i+1][j-1]:
                            ans += 1
                            dp[i][j] = True
        return ans
    
    # LC.516.最长回文子序列
    '''
        1. dp数组定义
            dp[i][j] 下标在[i..j]之间的子序列的最大长度
        2. 状态转移方程
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        3. 初始化
            if i == j: dp[i][j] = 1
    '''
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        # init
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        # dp
        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return max(map(max, dp))