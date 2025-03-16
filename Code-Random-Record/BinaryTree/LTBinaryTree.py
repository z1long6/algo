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
    
    # LT.116.填充每个节点的下一个右节点
    def connect(self, root: 'Optional[TreeNode]') -> 'Optional[TreeNode]':
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
        for index, _ in enumerate(res):
            len_ = len(_)
            for j, item in enumerate(_):
                if j == len_-1:
                    item.next = None
                else:
                    item.next = _[j+1]
        return root
    
    # LT.116.填充每个节点的下一个右节点Ⅱ
    def connect(self, root: 'Optional[TreeNode]') -> 'Optional[TreeNode]':
        if not root:
            return [None]
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
                level.append(node)
            res.append(level)
        for index, _ in enumerate(res):
            len_ = len(_)
            for j, item in enumerate(_):
                if j == len_-1:
                    item.next = None
                else:
                    item.next = _[j+1]
        return root
    
    # LT.104.二叉树的最大深度
    # recursion methods
    def maxDepth(root: Optional[TreeNode]) -> int:
        # top-down
        # use nonlocal keywords
        # ans = 0
        # def dfs(root: TreeNode, depth):
        #     # condition of stopping recursion
        #     if root is None:
        #         return
        #     depth += 1
        #     nonlocal ans
        #     ans = max(depth, ans)

        #     # logic of recursion
        #     dfs(root.left, depth)
        #     dfs(root.right, depth)
        
        # dfs(root, 0)

        # return ans
    
        # bottom-top
        if root is None:
            return 0
        left_depth = LTSolution.maxDepth(root.left)
        right_depth = LTSolution.maxDepth(root.right)
        return max(left_depth, right_depth) + 1

    # LT.111.二叉树的最小深度
    def minDepth(self, root: Optional[TreeNode]) -> int:         

        ans = int('inf')

        def getDepth(root: TreeNode, depth):

            # stop recursion
            if root is None:
                return 0
            
            depth += 1
            
            if root.left is None and root.right is None:
                nonlocal ans
                ans = min(ans, depth)

            getDepth(root.left, depth)
            getDepth(root.right, depth)
        
        getDepth(root, 0)
        return ans