class PrefixArray:
    '''
        数组的前缀和 prefix sum
    '''
    def __init__(self, nums: list[int]):
        s = [0] * (len(nums)+1)
        for i, x in enumerate(nums):
            s[i+1] = s[i] + x
        self.s = s

    def prefixSum(self, left: int, right: int) -> int:
        return self.s[right+1] - self.s[left]