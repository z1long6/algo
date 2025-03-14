import TreeNode
from collections import deque
class Solution:
    # traversal of recursion
    # root first order
    def firstOrder(root: TreeNode) -> list:
        res = []

        def traversal(node: TreeNode) -> None:
            if node is None:
                return
            res.append(node)
            traversal(node.left)
            traversal(node.right)
        
        traversal(root)
        return res
    
    # root second order
    def secondOrder(root: TreeNode) -> list:
        res = []

        def traversal(node: TreeNode) -> None:
            if node is None:
                return
            traversal(node.left)
            res.append(node)
            traversal(node.right)
        
        traversal(root)
        return res
    
    # root last order
    def lastOrder(root: TreeNode) -> list:
        res = []

        def traversal(node: TreeNode) -> None:
            if node is None:
                return
            traversal(node.left)
            traversal(node.right)
            res.append(node)
        
        traversal(root)
        return res
    
    # traversal of non-recursion
    def preOrder(root: TreeNode) -> list:
        if not root:
            return None
        stack = [root]
        res = []
        while stack is not None:
            temp_node = stack.pop()
            res.append(temp_node)
            if temp_node.right:
                stack.append(temp_node.right)
            if temp_node.left:
                stack.append(temp_node.left)
        return res
    
    def lastOrder(root: TreeNode) -> list:
        if not root:
            return None
        stack = [root]
        res = []
        while stack:
            temp_node = stack.pop()
            res.append(temp_node)
            if temp_node.left:
                stack.append(temp_node.left)
            if temp_node.right:
                stack.append(temp_node.right)
        return res[::-1] # reverse the list
    
    def secondOrder(root: TreeNode) -> list:
        if not root:
            return None
        stack = []
        res = []
        cur = root
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else: # to the bottom on the left direction, to the right
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right
        return res
    
    # the union way of traveral of the binary tree
    # second order in the union way
    def unionSecondOrder(root: TreeNode) -> list:
        if not root:
            return None
        stack = [(root, False)]
        result = []
        while stack:
            node, visited = stack.pop()
            if visited:
                result.append(node.val)
            else:
                # second order
                # the sequence of traversal is:  left - middle - right
                # the sequence of in-stack: right - middle - left
                # if we want different order of traversal, adjust the sequence of the methods of append
                if node.right:
                    stack.append((node.right, False))

                stack.append((node, True))

                if node.left:
                    stack.append((node.left, False))

    # BFS in Binary Tree of non-recursion
    def BFSofBinaryTree(root: TreeNode) -> list:
        if not root:
            return []
        queue_ = deque()
        queue_.append(root)
        res = []
        while queue_:
            level = []
            for _ in range(len(queue_)):
                node = queue_.popleft()
                level.append(node.val)
                if node.left:
                    queue_.append(node.left)
                if node.right: 
                    queue_.append(node.right)
            res.append(level)
        return res
    
    # BFS in BInary Tree of recursion
    def traversal(node: TreeNode, res: list, level: int):
        # stop recursion
        if not node:
            return []

        # handle the recursion
        if len(res) < level:
            res.append([])
        
        res[level-1].append(node.val)

        Solution.traversal(node.left, res, level+1)
        Solution.traversal(node.right, res, level+1)

        return res
    
    def levelOrderRecursion(root: TreeNode):
        level = 1
        res = []
        # parameter of recursion
        return Solution.traversal(root, res, level)