# How to work out the problem of backtracing?
## 1. 组合问题
> 回溯的本质仍然是递归 
考虑递归三部曲:  
1. 递归传递参数和返回值 backtracing() -> None:
2. 递归的结束条件
3. 在每次递归中需要执行的逻辑
---
### 1.1 分析回溯问题
![''](../assert/backtracing1.png "picture 1")
以最简单的组合问题为例，在**抽象树**的每一层中是遍历每种可能取值，而在树枝（深度）中遍历则是寻找每种组合的子组合

### 1.2 回溯问题的模版代码
仍然组合问题为例
```python
# 组合问题
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
    
    for i in range(startindex, n - (k - len(path)) + 2): # 剪枝
        path.append(i)
        self.backtracing(n, k, i+1, path, result)
        path.pop()
```

### 1.3 什么时候需要startindex
每次在集合中选取元素，可选择的范围随着选取的进行而收缩（可选取元素减少），调整可选取的范围，就是要靠`startindex`

### 1.4 在for - 层遍历过程中添加额外判断逻辑
以`Leetcode`第40题组合总和为例，需要对元素进行去重
```python
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
```
在本题中，同一个解可以出现同一元素（按树枝搜索），但在解集合中不能出现相同元素，如`[[1,1,2],[1,2,1]]`是不合法的，使用`candidates[]`与`used[]`数组来控制按层搜索和按树枝搜索时出现重复元素是否可以被添加到解集合中，体现在代码中：
```python
for i in range(startindex, len(candidates)):
        if i > 0 and candidates[i-1] == candidates[i] and used[i-1] == False:
            continue
```