from TreeNode import TreeNode
from BinaryTreeBase import Solution
from LTBinaryTree import LTSolution

def createTree():
    root = TreeNode(0)
    # root.left = TreeNode(2)
    # root.right = TreeNode(3)
    node = root
    for i in range(1, 5):
        node.left = TreeNode(i)
        node = node.left
    return root

if __name__ == '__main__':
    root = createTree()
    print(LTSolution.maxDepth(root))
    # for i, item in enumerate(LTSolution.maxDepth(None, root)):
    #     print(item)