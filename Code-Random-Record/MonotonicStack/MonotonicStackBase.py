class Solution:
    '''
        何时使用单调栈:
            在一维数组中, 求任一元素的左边或右边第一个比自己大或小的元素
    '''
    # LC.749.每日温度
    # 从右向左遍历元素
    def dailyTemperatures_0(self, temperatures: list[int]) -> list[int]:
        n = len(temperatures)
        res = [0] * n
        st = []
        for i in range(n-1, -1, -1):
            st.append(i)
            # 维持单调
            while st and temperatures[i] >= temperatures[st[-1]]:
                st.pop()
            # 计算
            if st:
                res[i] = st[-1] - i 
        return res
    
    # 从左向右遍历元素
    '''
        栈中只保存尚未找到结果的元素, 且呈递增趋势
    '''
    def dailyTemperatures_1(self, temperatures: list[int]) -> list[int]:
        res = [0] * len(temperatures)
        st = []
        for i, t in enumerate(temperatures):
            while st and t > temperatures[st[-1]]:
                j = st.pop()
                res[j] = i - j
            st.append(i)
        return res
    
    # LC.496.下一个更大元素1
    def nextGreaterElement_0(self, nums1: list[int], nums2: list[int]) -> list[int]:
        res = [-1] * len(nums1)
        for i, x in enumerate(nums1):
            temp = float('inf')
            for j, y in enumerate(nums2):
                if x == y:
                    temp = x
                    continue
                if y > temp:
                    res[i] = y
                    break
        return res
    
    # LC.503.下一个更大元素2
    def nextGreaterElements_1(self, nums: list[int]) -> list[int]:
        n = len(nums)
        ans = [-1] * n
        for i, x in enumerate(nums):
            j = i + 1
            for k in range(n-1):
                if j >= n:
                    j = j % n
                if nums[j] > x:
                    ans[i] = nums[j]
                    break
                j += 1
        return ans
    
    # LC.42.接雨水
    '''
        法1: 考虑将雨水面积和台阶面积求和, 再减去台阶面积即得到雨水面积
    '''
    def trap_0(self, height: list[int]) -> int:
        l = 0
        r = len(height) - 1
        sum_ = 0
        # while l <= r and h <= max(height):
        for h in range(1, max(height)+1):
            while height[l] < h:
                l += 1
            while height[r] < h:
                r -= 1
            sum_ += r - l + 1
            h += 1
        return sum_ - sum(height)

    '''
        法2: 横着计算水的面积
    '''
    def trap_1(self, height: list[int]) -> int:
        ans = 0
        st = []
        for i, x in enumerate(height):
            while st and x > height[st[-1]]:
                temp = st.pop()
                if not st:
                    break
                left = st[-1] # 在计算面积时, 至少需要三根柱子才能计算面积
                ans += (i - left - 1) * (min(height[st[-1]], x) - height[temp])
            st.append(i)
        return ans
    
    # LC.84.柱状图中的最大面积
    '''
        left 所选矩形左边界第一个比左边界小的元素
        right 所选矩形右边界第一个比右边界大的元素
    '''
    def largestRectangleArea(self, heights: list[int]) -> int:
        ans = 0
        left,right = [-1] * len(heights), [len(heights)] * len(heights)
        st = []
        # 记录left
        for i, x in enumerate(heights):
            while st and x <= heights[st[-1]]:
                st.pop()
            if st:
                left[i] = st[-1]
            st.append(i)
        st.clear()
        # 记录right
        for j in range(len(heights)-1, -1, -1):
            while st and heights[j] <= heights[st[-1]]:
                st.pop()
            if st:
                right[j] = st[-1]
            st.append(j)
        # 枚举heights数组中的每一个元素, 因为所选去的矩形的高度一定是数组中的元素
        for height, l, r in zip(heights, left, right):
            ans = max(ans, height * ((r-1) - (l+1) + 1))
        return ans
