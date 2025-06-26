'''
    This is Backtracing algorithm
'''
class Solutions:
    # LC.77.组合
    def combine(self, n: int, k: int) -> list[list[int]]:
        result = []
        self.backtracing(n, k, 1, [], result)
        return result

    # paragram
    def backtracing(self, n, k, startindex, path: list, result: list):
        # stop condition
        if len(path) == k:
            result.append(path[:])
            return
        
        # for i in range(startindex, n+1):
        #     path.append(i)
        #     self.backtracing(n, k, i+1, path, result)
        #     path.pop()
        for i in range(startindex, n - (k - len(path)) + 2): # 剪枝
            path.append(i)
            self.backtracing(n, k, i+1, path, result)
            path.pop()
    
    # LC.216.组合总和3
    def combinationSum3(self, k: int, n: int) -> list[list[int]]:
        path = []
        result = []

        def backtracing(n, k, start):
            if sum(path) > n:
                return
        
            if len(path) == k:
                if sum(path) == n:
                    result.append(path.copy())
                return
            
            for i in range(start, 10):

                path.append(i)
                backtracing(n, k, i+1)
                path.pop()

        backtracing(n, k, 1)
        return result

    # LC.17.电话号码的字母组合
    def letterCombinations(self, digits: str) -> list[str]:
        int(digits)

        str_dict = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]

        if len(digits) == 0:
            return []
        
        str_ = []
        result = []

        def backtracing(j: int) -> None:
            if len(str_) == len(digits):
                result.append(''.join(str_))
                return 
            
            for i in range(len(str_dict[int(digits[j])])):
                str_.append(str_dict[int(digits[j])][i])
                backtracing(j+1)
                str_.pop()
        
        backtracing(0)
        return result
    
    # LC.39.组合总和3
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        temp, res = [], []

        def backtracing(startindex: int) -> None:
            
            if target == sum(temp):
                res.append(temp.copy())
                return
            
            elif target < sum(temp):
                return
                

            for i in range(startindex, len(candidates)):
                temp.append(candidates[i])
                backtracing(i)
                temp.pop()
        
        backtracing(0)
        # 暴力去重
        # res = [list(item) for item in {tuple(sorted(sublist)) for sublist in res}]

        return res
    
    # LC.40.组合总和2
    def combinationSum2(self, candidates: list[int], target: int) -> list[list[int]]:
        candidates = sorted(candidates)
        temp, res = [], []

        def backtracing(startindex: int, used: list) -> None:
            
            if target == sum(temp):
                res.append(temp.copy())
                return
            
            elif target < sum(temp):
                return
                

            for i in range(startindex, len(candidates)):
                if i > 0 and candidates[i-1] == candidates[i] and used[i-1] == False:
                    continue

                temp.append(candidates[i])
                used[i] = True
                backtracing(i+1, used)
                used[i] = False
                temp.pop()
        
        backtracing(0, [False for i in range(len(candidates))])
        # 暴力去重
        # res = [list(item) for item in {tuple(sorted(sublist)) for sublist in res}]

        return res
    
    # LC.141.分割回文串
    def partition(self, s: str) -> list[list[str]]:
        res, path = [], []

        def isPalindrome(str_: str, startindex: int, endindex: int) -> bool:
            while(startindex < endindex):
                if str_[startindex] != str_[endindex]:
                    return False
                startindex += 1
                endindex -= 1
            return True

        def backtracing(startindex: int) -> None:
            
            if startindex == len(s):
                res.append(path.copy())
                return
            
            for i in range(startindex, len(s)):

                if isPalindrome(s, startindex, i):
                    path.append(s[startindex:i+1])
                else:
                    continue

                backtracing(i+1)
                path.pop()

        backtracing(0)
        return res
    
    # LC.93.复原IP地址
    def restoreIpAddresses(self, s: str) -> list[str]:
        if len(s) > 12:
            return []
        
        res = []
        temp_str = []

        def judgeValidIp(str_: str) -> bool:
            if str_.startswith('0') and len(str_) > 1:
                return False
            elif len(str_) > 3:
                return False
            elif int(str_) > 255:
                return False

            return True

        def backtracing(start_index: int) -> None:

            # 收集叶子节点
            if start_index == len(s) and len(temp_str) == 4:
                res.append('.'.join(temp_str))
            
            # 回溯逻辑
            for i in range(start_index, len(s)):
               
               # 判断本次截取是否满足合法ip段
               if not judgeValidIp(s[start_index:i+1]):
                    continue
               
               temp_str.append(s[start_index:i+1])
               backtracing(i+1)
               temp_str.pop()



        backtracing(0)
        return res
    
    # LC.78.子集
    # 子集问题需要收集抽象结构树上的所有节点
    def subsets(self, nums: list[int]) -> list[list[int]]:
        res, path = [], []

        # 回溯
        def backtracing(startindex: int) -> None:
            # 每次调用backtracing都是在遍历该树的一个节点
            res.append(path.copy())

            if startindex == len(nums):
                return
            
            for i in range(startindex, len(nums)):
                path.append(nums[i])
                backtracing(i+1)
                path.pop()
        
        backtracing(0)
        return res
    
    # LC.90.子集2
    # 去重问题，同一树层不能重复（不同的解集合），但在同一树枝可以重复（寻找唯一子集的过程，树的遍历深度加深）
    def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        res, path = [], []


        # 回溯
        def backtracing(startindex: int, used: list[bool]) -> None:
            # 每次调用backtracing都是在遍历该树的一个节点
            res.append(path.copy())

            if startindex == len(nums):
                return
            
            for i in range(startindex, len(nums)):
                # 去重
                if i > 0 and nums[i-1] == nums[i] and used[i-1] is False:
                    continue

                path.append(nums[i])
                used[i] = True
                backtracing(i+1, used)
                # 回溯
                used[i] = False
                path.pop()
        
        backtracing(0, [False for i in range(len(nums))])
        return res

    # LC.491.非递减子序列
    def findSubsequences(self, nums: list[int]) -> list[list[int]]:
        res, path = [], []

        # 回溯
        def backtracing(startindex: int) -> None:
            # 每次调用backtracing都是在遍历该树的一个节点

            # 收集每个节点
            if len(path) > 1:
                res.append(path.copy())

            # 到达叶子节点或层遍历完成
            if startindex == len(nums):
                return
            
            uset = set()

            for i in range(startindex, len(nums)):
                
                # 去重
                # uset记录同一层是否使用了相同的数字，path记录当前树枝按深度搜索是否符合要求
                if (path and path[-1] > nums[i]) or nums[i] in uset:
                    continue

                uset.add(nums[i])

                path.append(nums[i])
                backtracing(i+1)
                # 回溯
                path.pop()
        
        backtracing(0)
        return res
    
    # LC.46.全排列
    def permute(self, nums: list[int]) -> list[list[int]]:
        res, path = [], []

        def backtracing() -> None:
            if len(path) is len(nums):
                res.append(path.copy())
                return
            
            for i in range(len(nums)):
                # 数字不重复
                if nums[i] in path:
                    continue

                path.append(nums[i])
                backtracing()
                path.pop()

        backtracing()
        return res
    
    # LC.47.全排列2
    def permuteUnique(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        res, path = [], []

        def backtracing(used: list) -> None:
            if len(path) is len(nums):
                res.append(path.copy())
                return
            
            for i in range(len(nums)):
                if (i > 0 and nums[i-1] == nums[i] and not used[i-1]):
                    continue
                if used[i] == False: # 同一树枝i没使用过
                    used[i] = True
                    path.append(nums[i])
                    backtracing(used)
                    path.pop()
                    used[i] = False

        backtracing([False for i in range(len(nums))])
        return res
    
    # LC.332.重新安排行程
    def findItinerary(self, tickets: list[list[str]]) -> list[str]:
        pass
    