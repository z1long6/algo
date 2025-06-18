from typing import Optional
from TreeNode import TreeNode
from collections import deque
from BinaryTreeBase import Solution
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
    
    # LT.226.反转二叉树
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # if root is None:
        #     return None

        # res = Solution.BFSofBinaryTree(root)
        # for index in range(1, len(res)):
        #     list_ = list(reversed(res[index]))
        #     res[index] = list_
            # for j in range(len(res[index-1])):

            #     res[index-1][j].left = list_[2*j]

            #     res[index-1][j].right = list_[2*j+1]
        
        # return root

        # recursion 
        if root is None:
            return root
        
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)

        root.left, root.right = right, left
        return root
    
    # LT.101.对称二叉树
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        # recursion
        # if root is None:
        #     return True
        
        # def judge(left, right) -> bool:
        #     # stop recursion
        #     if not left and right:
        #         return False
        #     elif left and not right:
        #         return False
        #     elif not left and not right:
        #         return True
        #     elif left.val != right.val:
        #         return False
            
        #     # logic of recursion
        #     L = judge(left.left, right.right)
        #     R = judge(left.right, right.left)
        #     return L & R
        
        # return judge(root.left, root.right)

        # non-recursion
        if not root:
            return True
         
        queue_ = deque()
        queue_.append(root.left)
        queue_.append(root.right)

        while queue_:

            node1 = queue_.popleft()
            node2 = queue_.popleft()
            # node1 and node2 is None
            if not node1 and not node2:
                continue
            
            # node1 or node2 is None but they are different
            if node1 and not node2:
                return False
            elif not node1 and node2:
                return False
            elif node1.val != node2.val:
                return False
            
            # node1 and node2 are not none
            queue_.append(node1.left)
            queue_.append(node2.right)
            queue_.append(node1.right)
            queue_.append(node2.left)
        
        return True
    
    # LT.222.完全二叉树的节点个数
    def countNodes(self, root: Optional[TreeNode]) -> int:
        ans = 0
        def dfs(root: TreeNode) -> int:
            if root is None:
                return 0
            nonlocal ans
            ans += 1
            dfs(root.left)
            dfs(root.right)
            return ans
        dfs(root)
        return ans
        # use complete binary tree

    # LT.110.平衡二叉树
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def getDepth(root: TreeNode) -> int:
            if root is None:
                return 0
            ldepth = getDepth(root.left)
            if ldepth == -1: return -1 # early break
            rdepth = getDepth(root.right)
            if ldepth == -1 or rdepth == -1 or abs(ldepth - rdepth) > 1:
                return -1
            return max(ldepth, rdepth) + 1
        return True if getDepth(root) != -1 else False
    
    # LT.257.二叉树的所有路径
    def binaryTreePaths(self, root: Optional[TreeNode]) -> list[str]:
        # back up
        # if root is None:
        #     return []
        
        # stack = []
        # res = []
        # def getPaths(root: TreeNode):
        #     if root is None:
        #         return
            
        #     stack.append(str(root.val))

        #     if root.left == root.right:
        #         res.append('->'.join(stack))
        #     else:
        #         getPaths(root.left)
        #         getPaths(root.right)
            
        #     stack.pop()

        # getPaths(root)
        # return res

        # recursion
        res = []
        def getPath(root: TreeNode, path_: str):
            # node is none
            if root is None:
                return

            # node is not none
            path_ += str(root.val)

            # node is leaf
            if root.left == root.right:
                res.append(path_)
                return 
            
            path_ += '->'

            getPath(root.left, path_)
            getPath(root.right, path_)

        getPath(root, '')
        return res
    
    # LT.404.左叶子之和
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:

        res = []
        def firstOrder(root: TreeNode):
            if root is None:
                return
            # root
            if root.left != None and root.left.left == root.left.right:
                res.append(root.left.val)

            # left
            firstOrder(root.left)
            # right
            firstOrder(root.right)
        
        firstOrder(root)
        return sum(res)
    
    # LT.513.找到树左下角值
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        # level order
        # res = []

        # queue_ = deque()
        # queue_.append(root)
        # while queue_:
        #     level = []
        #     for i in range(len(queue_)):
        #         node = queue_.popleft()

        #         level.append(node.val)

        #         if node.left:
        #             queue_.append(node.left)
        #         if node.right:
        #             queue_.append(node.right)

        #     res.append(level)
        
        # return res[-1][0]
        self.maxDepth_ = -1
        self.res = None

        # recursion
        def depth(root: TreeNode, depth_: int):

            if root.left == root.right: # leaf node
                if depth_ > self.maxDepth_: # depth is higher
                    self.maxDepth_ = depth_
                    self.res = root.val
                return
            
            if root.left:
                depth_ += 1
                depth(root.left, depth_)
                depth_ -=1

            if root.right:
                depth_ +=1
                depth(root.right, depth_)
                depth_ -=1

        depth(root, 0)

        return self.res

    # LT.112.路径总和
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        # backtrack
        # sum_ = []
        # flag = False
        # def getDepth(root: TreeNode):
        #     if root is None:
        #         return
             
        #     sum_.append(root.val)

        #     if root.left is root.right:
        #         if sum(sum_) == targetSum:
        #             nonlocal flag
        #             flag = True
        #     else:
        #         getDepth(root.left)
        #         getDepth(root.right)  
            
        #     sum_.pop()
        #     return
        
        # getDepth(root)
        # return flag

        flag = False
        # recursion
        def getDepth(root: TreeNode, sum_) -> bool:
            if root is None:
                return 
            
            sum_ += root.val

            if root.left == root.right:
                if sum_ == targetSum:
                    nonlocal flag
                    flag = True

            getDepth(root.left, sum_)
            getDepth(root.right, sum_)
        
        getDepth(root, 0)
        return flag
    
    # LT.106.从中序和后序来创建二叉树
    def buildTree(self, inorder: list[int], postorder: list[int]) -> Optional[TreeNode]:

        # define the recursion function of create a binary tree
        def createBinaryTreeFromOrder(root: TreeNode, left_: list[int], right_: list[int]):
            # stop condition
            if len(left_) == 0: # none node
                return
            
            # recursion logic
            new_root = TreeNode(right_[-1])

            tempi = -1
            for i, x in enumerate(left_):
                if x == right_[-1]:
                    tempi = i
                    break

            new_root.left = createBinaryTreeFromOrder(new_root, left_[:tempi], right_[:len(left_[:tempi])]) # left of tempi
            new_root.right = createBinaryTreeFromOrder(new_root, left_[tempi+1:], right_[len(left_[:tempi]):-1]) # right of tempi
            
            return new_root
        
        root = createBinaryTreeFromOrder(None, inorder, postorder)
        return root
    
    # LT.654.最大二叉树
    def constructMaximumBinaryTree(self, nums: list[int]) -> Optional[TreeNode]:
        
        root = None

        if len(nums) == 0:
            return None
        
        # recursion logic
        index = nums.index(max(nums))
        root = TreeNode(nums[index])

        root.left = self.constructMaximumBinaryTree(nums[:index]) 
        root.right = self.constructMaximumBinaryTree(nums[index+1:])

        return root
    
    # LC.617.合并二叉树
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if root1 == None:
            return root2
        if root2 == None:
            return root1
        root1.val += root2.val
        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.right = self.mergeTrees(root1.right, root2.right)
        return root1
    
    # LC.700.二叉搜索树中的搜索 
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if root is None: 
            return None
        if root.val == val: 
            return root
        
        if root.val < val:
            return self.searchBST(root.right, val)
        elif root.val > val:
            return self.searchBST(root.left, val)

    # LC.98.验证二叉搜索树
    # def isValidBST(self, root: Optional[TreeNode]) -> bool:
    #     return self.judgeBST(root, float('-inf'), float('inf'))

    # # pre order in search binary tree
    # def judgeBST(self, root, lower, higher):
    #     if root is None:
    #         return True

    #     x = root.val
        
    #     return lower < x < higher and \
    #         self.judgeBST(root.left, lower, root.val) and \
    #         self.judgeBST(root.right, root.val, higher)
    # in order
    pre = float('-inf')
    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        if root is None:
            return True
        
        if not self.isValidBST(root.left):
            return False
        
        if root.val <= self.pre:
            return False
        
        self.pre = root.val
        
        return self.isValidBST(root.right)
    
    # LC.530.二叉搜索树的最小绝对差
    # def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
    #     result = self.firstOrder(root)
    #     minimun = float('inf')
    #     for i in range(1, len(result)):
    #         for j in range(0, i-1):
    #             if  abs(result[i] - result[j]) < minimun:
    #                 minimun = abs(result[i] - result[j])
    #     return minimun
    
    # def firstOrder(self, root: TreeNode) -> list:
    #     res = []

    #     def traversal(node: TreeNode) -> None:
    #         if node is None:
    #             return
    #         res.append(node.val)
    #         traversal(node.left)
    #         traversal(node.right)
        
    #     traversal(root)
    #     return res     
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        ans = float('inf')
        pre = float('-inf')

        def inOrder(root):
            if root is None:
                return
            inOrder(root.left)
            nonlocal ans, pre
            ans = min(ans, root.val - pre)
            pre = root.val
            inOrder(root.right)

        inOrder(root)        
        return ans
    
    # LC.501.二叉搜索树中的众数
    def findMode(self, root: Optional[TreeNode]) -> list[int]:
        map = {}
        res = []
        pre = float('inf')

        def inOrder(root):
            if root is None:
                return
            inOrder(root.left)
            nonlocal pre, map
            if pre == root.val:
                map[root.val] += 1
            else:
                map[root.val] = 1    
            pre = root.val
            inOrder(root.right)
        
        inOrder(root)
        max_value = max(map.values())
        for key, item in map.items():
            if item == max_value:
                res.append(key)
        return res
    
    # LC.236.二叉树的最近公共祖先
    # 回溯方法
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        path1, path2 = [], []


        def findPath(root, target, path, result):
            if not root:
                return
            
            path.append(root)
            if root == target:
                result.extend(path.copy())
                return
            
            findPath(root.left, target, path, result)
            findPath(root.right, target, path, result)

            path.pop()
        
        findPath(root, p, [], path1)
        findPath(root, q, [], path2)

        for i in range(len(path1)-1, -1, -1):
            for j in range(len(path2)-1, -1, -1):
                if path2[j] == path1[i]:
                    return path1[i]
        
        return None
    
    # 二叉树
    def lowestCommonAncestor2(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None or root is p or root is q:
            return root
        
        left = self.lowestCommonAncestor2(root.left, p, q)
        right = self.lowestCommonAncestor2(root.right, p, q)

        if left and right:
            return root
        
        if left:
            return left
        
        return right
    
    # 二叉搜索树
    def lowestCommonAncestor3(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        x = root.val
        if p.val < x and q.val < x:
            return self.lowestCommonAncestor3(root.left, p, q)
        if p.val > x and q.val > x:
            return self.lowestCommonAncestor3(root.right, p, q)
        return root
    
    # LC.701.二叉搜索树中的插入值
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if root is None:
            return TreeNode(val)
        res = root
        self.dfs701(root, val)
        return res

    def dfs701(self, root, val):
        if root is None:
            return 

        if root.val > val:
            if root.left is None:
                newNode =  TreeNode(val, None, None)
                root.left = newNode
                return
            self.dfs701(root.left, val)

        if root.val < val:
            if root.right is None:
                newNode =  TreeNode(val, None, None)
                root.right = newNode
                return
            self.dfs701(root.right, val)

    # LC.450.删除二叉搜索树中的节点
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if root is None:
            return root
        
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        elif root.left is None or root.right is None:
            root = root.left if root.left else root.right
        else:
            next_node = root.right
            while next_node.left:
                next_node = next_node.left
            next_node.left = root.left
            next_node.right = self.deleteNode(root.right, next_node.val)
            return next_node
        return root
    
    # LC.669.修剪二叉树
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        if root is None:
            return root

        if root.val < low:
            return self.trimBST(root.right, low, high)
        
        if root.val > high:
            return self.trimBST(root.left, low, high)
        
        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)
        return root

    # LC.108.将有序数组转换成(平衡)二叉搜索树
    def sortedArrayToBST(self, nums: list[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        m = len(nums) // 2
        left = self.sortedArrayToBST(nums[:m])
        right = self.sortedArrayToBST(nums[m+1:])
        return TreeNode(nums[m], left, right)
    
    # LC.238.把二叉搜索树转换为累加树
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        pre = 0

        def traversal238(root):
            if root is None:
                return None
            nonlocal pre
            traversal238(root.rigth)
            root.val += pre
            pre = root.val
            traversal238(root.left)
        
        traversal238(root)
        return root