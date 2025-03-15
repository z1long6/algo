from TreeNode import TreeNode
from BinaryTreeBase import Solution
from LTBinaryTree import LTSolution

def createTree():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    return root

if __name__ == '__main__':
    root = createTree()
    for i, item in enumerate(LTSolution.rightSideView(None, root)):
        print(item)