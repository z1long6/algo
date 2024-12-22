class Solution:
    def generateMatrix(self, n: int) -> list[list[int]]:
        nums = [[0] * n for _ in range(n)]
        startx, starty = 0, 0
        loop = n // 2 # 迭代次数
        mid = n // 2 # 奇数矩阵 中心点
        count = 1 
        for offset in range(1, loop+1):
            # 行：从左->右
            for j in range(starty, n-offset):
                nums[startx][j] = count
                count += 1
            # 列：从上->下
            for i in range(startx, n-offset):
                nums[i][n-offset] = count
                count += 1
            # 行：从右->左
            for j in range(n-offset, starty, -1):
                nums[n-offset][j] = count
                count += 1
            # 列：从下->上
            for i in range(n-offset, startx, -1):
                nums[i][starty] = count
                count += 1
            startx += 1
            starty += 1
        # 填充中心点
        if (n % 2) == 1:
            nums[mid][mid] = n ** 2
        return nums