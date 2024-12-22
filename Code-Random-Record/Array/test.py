# test function
import BinarySearch
import RemoveElement
import SquareOrderedArray
import SubarrayWithSmallestLength
import SpiralMartix
if __name__ == '__main__':
    # test case
    nums = [2,3,1,2,4,3]
    target = 7
    # BinarySearch
    # print(BinarySearch.Solution.search(None, nums=[-1, 0, 3, 5, 9, 12], target=9))
    print(BinarySearch.Solution.searchInsert(None, [1,3,5,6], 2))

    # RemoveElement
    # print(RemoveElement.Solution.removeElement3(None, nums, val))

    # SquareOrderedArray
    # print(SquareOrderedArray.Solution.sortedSquares2(None, [-7,-3,2,3,11]))

    # Subarray with smallest length
    # print(SubarrayWithSmallestLength.Solution.minSubArrayLen2(None, target, nums))

    # Spiral Matrix
    # print(SpiralMartix.Solution.generateMatrix(None, 3))