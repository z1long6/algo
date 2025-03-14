from TreeNode import TreeNode
from BinaryTreeBase import Solution

def createTree():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    return root

if __name__ == '__main__':
    root = createTree()
    for i, item in enumerate(Solution.levelOrderRecursion(root)):
        print(item)