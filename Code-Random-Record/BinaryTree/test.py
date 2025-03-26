from TreeNode import TreeNode
from BinaryTreeBase import Solution
from LTBinaryTree import LTSolution

def createTree():
    root = TreeNode(0)
    root.left = TreeNode(1)
    root.left.left = TreeNode(2)
    # root.right = TreeNode(7)
    # node1 = root.left
    # node2 = root.right

    # node1.left = TreeNode(1)
    # node1.right = TreeNode(3)

    # node2.left = TreeNode(6)
    # node2.right = TreeNode(9)
    # node = root
    # for i in range(1, 5):
    #     node.left = TreeNode(i)
    #     node = node.left
    return root

if __name__ == '__main__':
    root = createTree()
    my_solution = LTSolution() 
    print(my_solution.hasPathSum(root, 3))
    # for i, item in enumerate(LTSolution.maxDepth(None, root)):
    #     print(item)
    # lis_ = [4,2,7,1,3,6,9]
    # print(list(reversed(lis_)))