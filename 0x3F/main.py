# from SlidingWindow import Solution
from BinaryAlgo import Solution as Solution2

solution = Solution2()
# result = solution.longestAlternatingSubarray(
#     [3,2,5,4], 5
# )
result = solution.answerQueries(
    [4,5,2,1], [3,10,21]
)
print(result)