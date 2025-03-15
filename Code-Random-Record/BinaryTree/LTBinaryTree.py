from typing import Optional
from TreeNode import TreeNode
from collections import deque
class LTSolution:
    # LT.199.二叉树的右视图
    def rightSideView(self, root: TreeNode) -> list[int]:
        # non-recursion
        # if not root:
        #     return []
        # queue_ = deque()
        # res = []
        # queue_.append(root)
        # while queue_:
        #     level = []
        #     for _ in range(len(queue_)):
        #         node = queue_.popleft()
        #         if node.left:
        #             queue_.append(node.left)
        #         if node.right:
        #             queue_.append(node.right)
        #         level.append(node.val)
        #     res.append(level)
        # res2 = []
        # for index, list_ in enumerate(res):
        #     res2.append(list_[len(list_)-1])
        # return res2
        
        # recursion
        res = []
        def dfs(root: TreeNode, depth: int) -> None:
            if not root:
                return

            # operator in recursion
            if (len(res)+1) == depth:
                res.append(root.val)

            dfs(root.right, depth+1) # right tree
            dfs(root.left, depth+1) # left tree
        
        
        dfs(root, 1)
        return res
    
    # LT.515.在每个树行中找最大值
    def largestValues(self, root: Optional[TreeNode]) -> list[int]:
        if not root:
            return []
        queue_ = deque()
        res = []
        queue_.append(root)
        while queue_:
            level = []
            for _ in range(len(queue_)):
                node = queue_.popleft()
                if node.left:
                    queue_.append(node.left)
                if node.right:
                    queue_.append(node.right)
                level.append(node.val)
            res.append(level)
        res2 = []
        for index, _ in enumerate(res):
            res2.append(max(_))
        return res2