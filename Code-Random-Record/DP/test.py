from DPBase import Solution

if __name__ == '__main__':
    mySolution = Solution()
    s = 'leetcode'
    wordDict = ['leet', 'code']
    result = mySolution.wordBreak(s, wordDict)
    print(result)