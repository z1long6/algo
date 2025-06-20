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
    
    # LC.39.组合总和
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
    
    # LC.40.组合总和
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