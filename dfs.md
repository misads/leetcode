# DFS(深度优先搜索)

## A98. 验证二叉搜索树

难度`中等`

#### 题目描述

给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：

- 节点的左子树只包含**小于**当前节点的数。
- 节点的右子树只包含**大于**当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

> **示例 1:**

```
输入:
    2
   / \
  1   3
输出: true
```

> **示例 2:**

```
输入:
    5
   / \
  1   4
     / \
    3   6
输出: false
解释: 输入为: [5,1,4,null,null,3,6]。
     根节点的值为 5 ，但是其右子节点值为 4 。
```

#### 题目链接

<https://leetcode-cn.com/problems/validate-binary-search-tree/>

#### **思路:**

　　DFS，在递归过程中记录每棵子树的最小值和最大值，然后和根节点比较。  

#### **代码:**  

　　**写法一：**

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        
        def dfs(root):  # 返回最小值 最大值 以及是不是BST
            min_root = max_root = root.val
            bst_root = True
            if root.left:
                if root.left.val >= root.val: return 0, 0, False  # 快速判断 减少搜索次数

                min_left, max_left, bst_left = dfs(root.left)
                if not bst_left or max_left >= root.val:  # 左子树不是或者左子树最大值大于根结点
                    return 0, 0, False
                min_root = min(min_root, min_left)
                max_root = max(max_root, max_left)

            if root.right:
                if root.right.val <= root.val: return 0, 0, False
                
                min_right, max_right, bst_right = dfs(root.right)
                if not bst_right or min_right <= root.val:  # 左子树不是或者左子树最大值大于根结点
                    return 0, 0, False
                min_root = min(min_root, min_right)
                max_root = max(max_root, max_right)

            return min_root, max_root, bst_root

        if not root:
            return True

        _, _, ans = dfs(root)

        return ans

```

　　**写法二：**

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        
        def dfs(root, minimum, maximun):  # 当前节点的最小和最大范围
            if not root: return True

            if root.val <= minimum or root.val >= maximun:
                return False

            return dfs(root.left, minimum, root.val) and dfs(root.right, root.val, maximun)

        return dfs(root, float('-inf'), float('inf'))

```

## A99. 恢复二叉搜索树

难度`困难`

#### 题目描述

二叉搜索树中的两个节点被错误地交换。

请在不改变其结构的情况下，恢复这棵树。

> **示例 1:**

```
输入: [1,3,null,null,2]

   1
  /
 3
  \
   2

输出: [3,1,null,null,2]

   3
  /
 1
  \
   2
```

> **示例 2:**

```
输入: [3,1,4,null,null,2]

  3
 / \
1   4
   /
  2

输出: [2,1,4,null,null,3]

  2
 / \
1   4
   /
  3
```

**进阶:**

- 使用 O(*n*) 空间复杂度的解法很容易实现。
- 你能想出一个只使用常数空间的解决方案吗？

#### 题目链接

<https://leetcode-cn.com/problems/recover-binary-search-tree/>

#### **思路:**

　　二叉搜索树的中序遍历一定是有序的。  

　　先中序遍历，将遍历的结果排序，对比它们可以找到两个更改过的结点，交换它们即可。  　　　　

#### **代码:**  

```python
class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        nodes = []
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            nodes.append(node)  # 中根遍历
            dfs(node.right)

        dfs(root)

        s = sorted(nodes, key=lambda kv: kv.val)

        a = None
        for x, y in zip(nodes, s):
            if x.val != y.val:
                if a is None:
                    a = x
                else:
                    b = x

        a.val, b.val = b.val, a.val
         
```

## A100. 相同的树

难度`简单`

#### 题目描述

给定两个二叉树，编写一个函数来检验它们是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

> **示例 1:**

```
输入:       1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

输出: true
```

> **示例 2:**

```
输入:      1          1
          /           \
         2             2

        [1,2],     [1,null,2]

输出: false
```

> **示例 3:**

```
输入:       1         1
          / \       / \
         2   1     1   2

        [1,2,1],   [1,1,2]

输出: false
```

#### 题目链接

<https://leetcode-cn.com/problems/same-tree/>

#### **思路:**

　　先判断根结点是否相同，然后递归判断左右子树。  　　

#### **代码:**

```python
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:  # 都为空
            return True

        if not p and q or not q and p:  # 有一个为空 另一个不为空
            return False

        if p.val != q.val:  # 都不为空 但是值不同
            return False
            
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)  # 递归判断左右子树

```

## A101. 对称二叉树

难度`简单`

#### 题目描述

给定一个二叉树，检查它是否是镜像对称的。

例如，二叉树 `[1,2,2,3,4,4,3]` 是对称的。

```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```

但是下面这个 `[1,2,2,null,3,null,3]` 则不是镜像对称的:

```
    1
   / \
  2   2
   \   \
   3    3
```

#### 题目链接

<https://leetcode-cn.com/problems/symmetric-tree/>

#### **思路:**

　　双指针，`left`从左子树开始，`right`从右子树开始。`left`向左遍历时`right`就向右遍历；`left`向右遍历时`right`就向左遍历；如果有不相等就返回`False`。  

#### **代码:**

```python
lass Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def dfs(node_left, node_right):  # 双指针
            if not node_left and not node_right: 
                return True

            if not node_left and node_right or not node_right and node_left:
                return False

            if node_left.val != node_right.val:
                return False

            return dfs(node_left.left, node_right.right) and dfs(node_left.right, node_right.left) 

        if not root:
            return True

        return dfs(root.left, root.right)
```

## A104. 二叉树的最大深度

难度`简单`

#### 题目描述

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

**说明:** 叶子节点是指没有子节点的节点。

> **示例：**  

给定二叉树 `[3,9,20,null,null,15,7]`，

```
    3
   / \
  9  20
    /  \
   15   7
```

返回它的最大深度 3 。

#### 题目链接

<https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/>

#### **思路:**

　　方法一：dfs，用一个全局变量记录最大深度，如果当前结点的深度大于最大深度则更新最大深度。  

　　方法一：bfs，层序优先遍历，返回最后一层的是第几层。  　　　　

#### **代码:**

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        ans = 0
        def dfs(node, depth):
            nonlocal ans
            if not node:
                return
            depth += 1
            ans = max(ans, depth)
            if node.left:
                dfs(node.left, depth)
            
            if node.right:
                dfs(node.right, depth)

        dfs(root, 0)
        return ans
      
```

## A108. 将有序数组转换为二叉搜索树

难度`简单`

#### 题目描述

将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过 1。

> **示例:**

```
给定有序数组: [-10,-3,0,5,9],

一个可能的答案是：[0,-3,9,-10,null,5]，它可以表示下面这个高度平衡二叉搜索树：

      0
     / \
   -3   9
   /   /
 -10  5
```

#### 题目链接

<https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/>

#### **思路:**

　　模板题。见[有序数组构建平衡二叉树](/实用模板?id=有序数组构建平衡二叉树)。  

#### **代码:**

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def build(nums, i, j):
            mid = (i+j)//2
            root = TreeNode(nums[mid])
            if(i==j):
                return root
            if i <= mid-1:
                root.left = build(nums,i,mid-1)
            if mid+1 <= j:
                root.right = build(nums, mid+1, j)

            return root

        if not nums: return []
        return build(nums, 0, len(nums)-1)
```

## A110. 平衡二叉树

难度`简单`

#### 题目描述

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

> 一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过1。

> **示例 1:**

给定二叉树 `[3,9,20,null,null,15,7]`

```
    3
   / \
  9  20
    /  \
   15   7
```

返回 `true` 。

> **示例 2:**

给定二叉树 `[1,2,2,3,3,null,null,4,4]`

```
       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
```

返回 `false` 。

#### 题目链接

<https://leetcode-cn.com/problems/balanced-binary-tree/>

#### **思路:**

　　dfs，搜索时返回当前树的深度，以及是否平衡。  

　　左右子树的深度之差绝对值大于1，则该树不平衡。  

#### **代码:**

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def dfs(node):  # depth, 是否平衡
            if not node:
                return 0, True

            depth = 0

            left, left_balance = dfs(node.left) 
            depth = max(depth, left)

            right, right_balance = dfs(node.right) 
            depth = max(depth, right)

            if not left_balance or not right_balance or abs(left - right) > 1:
                return 0, False


            return depth + 1, True

        _, balance = dfs(root)
        return balance
```

## A111. 二叉树的最小深度

难度`简单`

#### 题目描述

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

**说明:** 叶子节点是指没有子节点的节点。

> **示例:**

给定二叉树 `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回它的最小深度  2.

#### 题目链接

<https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/>

#### **思路:**

　　dfs。用一个**全局变量**`ans`记录最小深度，如果遇到叶子结点的深度小于`ans`，就更新`ans`。  

#### **代码:**

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        ans = float('inf')
        def dfs(node, depth):
            nonlocal ans
            if not node:
                return 

            depth += 1
            if not node.left and not node.right:
                ans = min(ans, depth)
            else:
                dfs(node.left, depth)
                dfs(node.right, depth)
            
        if not root:
            return 0
            
        dfs(root, 0)
        return ans
```

## A112. 路径总和

难度`简单`

#### 题目描述

给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。

**说明:** 叶子节点是指没有子节点的节点。

> **示例:**   

给定如下二叉树，以及目标和 `sum = 22`，

```
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
```

返回 `true`, 因为存在目标和为 22 的根节点到叶子节点的路径 `5->4->11->2`。

#### 题目链接

<https://leetcode-cn.com/problems/path-sum/>

#### **思路:**

　　dfs。搜索到每个结点都加上这个结点的`val`，注意函数返回时要将加上的`val`减去。  

　　当某个叶子结点的和等于`sum`时返回`True`。  

#### **代码:**

```python
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        temp = 0
        def dfs(node):
            nonlocal temp
            if not node:
                return False

            temp += node.val

            if not node.left and not node.right and temp == sum:
                return True

            if dfs(node.left) or dfs(node.right):
                return True

            temp -= node.val
            return False

        return dfs(root)
```

## A113. 路径总和 II

难度`中等`

#### 题目描述

给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。

**说明:** 叶子节点是指没有子节点的节点。

> **示例:**  

给定如下二叉树，以及目标和 `sum = 22`，

```
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
```

返回:

```
[
   [5,4,11,2],
   [5,8,4,5]
]
```

#### 题目链接

<https://leetcode-cn.com/problems/path-sum-ii/>

#### **思路:**

　　比上一题[A112. 路径综合](/dfs?id=a112-路径总和)多了一步记录`从根结点到当前结点的路径`。  

#### **代码:**

```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        temp = 0
        path = []  # 记录路径
        ans = []
        def dfs(node):
            if not node:
                return 
            nonlocal temp

            temp += node.val
            path.append(node.val)
            pop_idx = len(path) - 1  # 记录插入到path中的位置 在函数返回前删除掉

            if not node.left and not node.right and temp == sum:
                ans.append(path.copy())
            else:
                dfs(node.left)
                dfs(node.right)

            # 还原到调用之前的状态
            temp -= node.val  
            path.pop(pop_idx)

        dfs(root)
        return ans
      
```

## A114. 二叉树展开为链表

难度`中等`

#### 题目描述

给定一个二叉树，[原地](https://baike.baidu.com/item/原地算法/8010757)将它展开为链表。

例如，给定二叉树

```
    1
   / \
  2   5
 / \   \
3   4   6
```

将其展开为：

```
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```

#### 题目链接

<https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/>

#### **思路:**

　　dfs。  规则如下：  

      ① 只有右子树：不做任何操作 对右子树递归
      ② 叶子结点：把自己返回回去
      ③ 只有左子树：左子树放到右子树 然后把左子树置空 对右子树递归
      ④ 左右子树都有：dfs(左子树).right = 右子树 node.right=左子树 然后把左子树置空 对(之前的)右子树递归
#### **代码:**

```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        # 只有右子树： 不做任何操作
        # 叶子结点：把自己返回回去
        # 只有左子树：左子树放到右子树 然后把左子树置空
        # 左右子树都有： dfs(左子树).right = 右子树 node.right=左子树 然后把左子树置空
        def dfs(node):
            if not node:
                return 

            if not node.left and not node.right:  # 叶子结点
                return node

            if node.left and node.right:
                right = node.right
                dfs(node.left).right = node.right
                node.right = node.left
                node.left = None
                return dfs(right)

            if node.left:
                left = node.left
                node.right = node.left
                node.left = None
                return dfs(left)

            if node.right:
                return dfs(node.right)

        dfs(root)
```

## A116. 填充每个节点的下一个右侧节点指针

难度`中等`

#### 题目描述

给定一个**完美二叉树**，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

```
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
```

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 `NULL`。

初始状态下，所有 next 指针都被设置为 `NULL`。

> **示例：**

<img src="_img/116.png" style="zoom:60%"/>

```
输入：{"$id":"1","left":{"$id":"2","left":{"$id":"3","left":null,"next":null,"right":null,"val":4},"next":null,"right":{"$id":"4","left":null,"next":null,"right":null,"val":5},"val":2},"next":null,"right":{"$id":"5","left":{"$id":"6","left":null,"next":null,"right":null,"val":6},"next":null,"right":{"$id":"7","left":null,"next":null,"right":null,"val":7},"val":3},"val":1}

输出：{"$id":"1","left":{"$id":"2","left":{"$id":"3","left":null,"next":{"$id":"4","left":null,"next":{"$id":"5","left":null,"next":{"$id":"6","left":null,"next":null,"right":null,"val":7},"right":null,"val":6},"right":null,"val":5},"right":null,"val":4},"next":{"$id":"7","left":{"$ref":"5"},"next":null,"right":{"$ref":"6"},"val":3},"right":{"$ref":"4"},"val":2},"next":null,"right":{"$ref":"7"},"val":1}

解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。 
```

**提示：**

- 你只能使用常量级额外空间。
- 使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。

#### 题目链接

<https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/>

#### **思路:**

　　填充后的`next`指针也可以使用上。  　　

#### **代码:**

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return None
            
        if not root.left:
            return root

        root.left.next = root.right
        if root.next:
            root.right.next = root.next.left

        self.connect(root.left)
        self.connect(root.right)

        return root

```

## A117. 填充每个节点的下一个右侧节点指针 II

难度`中等`

#### 题目描述

给定一个二叉树

```
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
```

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 `NULL`。

初始状态下，所有 next 指针都被设置为 `NULL`。 

**进阶：**

- 你只能使用常量级额外空间。
- 使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。

> **示例：**

<img src="_img/117.png" style="zoom:60%"/>

```
输入：root = [1,2,3,4,5,null,7]
输出：[1,#,2,3,#,4,5,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。
```

**提示：**

- 树中的节点数小于 `6000`
- `-100 <= node.val <= 100`

#### 题目链接

<https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node-ii/>

#### **思路:**

　　与上一题不同，不能直接用自身`next`的孩子作为孩子的`next`，因为自身的`next`可能没有孩子。如下图所示，当前结点为`4`，给左孩子`7`寻找`next`指针时，需要沿着`next`一直向右边扫描，一直到某个**有孩子的结点**时(`6`)才停下来。  

　　<img src="_img/a117.png" style="zoom:50%"/>

　　此外，要**先搜索右子树，后搜索左子树。**否则遇到如下图所示的情况时，给`7`的右孩子`0`找`next`时，`7`的`next的next`尚未搜索，因此无法找到`0`的`next`。  

　　<img src="_img/a117_2.png" style="zoom:50%"/>

#### **代码:**

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return None
            
        if not root.left and not root.right:
            return root

        curr = root.right
        if root.left and root.right:
            root.left.next = root.right
        elif root.left:
            curr = root.left

        temp = root.next
        next = None
        while temp:
            if temp.left:
                next = temp.left
                break
            elif temp.right:
                next = temp.right
                break
            temp = temp.next

        curr.next = next

        self.connect(root.right)
        self.connect(root.left)

        return root
```

## A124. 二叉树中的最大路径和

难度`困难`

#### 题目描述

给定一个**非空**二叉树，返回其最大路径和。

本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径**至少包含一个**节点，且不一定经过根节点。

> **示例 1:**

```
输入: [1,2,3]

       1
      / \
     2   3

输出: 6
```

> **示例 2:**

```
输入: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

输出: 42
```

#### 题目链接

<https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/>

#### **思路:**

　　dfs。搜索时要关注的有两种可能性：  

　　① 以根结点为**中间结点**(即穿过根结点)的最大路径，计算方法为`以左结点为终点的最大路径`+`以右结点为终点的最大路径`+`根结点值`。

　　② 以根结点为**终点**的最大路径，计算方法为 max(`以左结点为终点的最大路径`+`根结点值`，`以右结点为终点的最大路径`+`根结点值`，`单独根结点的值`)。简化后的表达式如下图所示：  

　　　　<img src="_img/a124.png" style="zoom:45%"/>　　  
　　后序遍历，访问根结点时已经知道了两个孩子结点的①和②，按条件递归即可。  

#### **代码:**

```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        ans = float('-inf')  

        def dfs(node): 
            if not node:
                return 0, 0

            nonlocal ans
            if not node.left and not node.right:  # 叶子结点
                ans = max(ans, node.val)
                return node.val, node.val

            cross_l, end_l = dfs(node.left)
            cross_r, end_r = dfs(node.right)

            cross_node = end_l + end_r + node.val
            end_node = max(end_l, end_r, 0) + node.val  # 可能用左子树或右子树，也可能都不用

            ans = max(ans, cross_node, end_node)  
            # cross_node表示以当前结点为中间结点的最大路径 
            # end_node表示以当前结点为终点的最大路径
            return cross_node, end_node

        dfs(root)
        return ans
      
```

## A129. 求根到叶子节点数字之和

难度`中等`

#### 题目描述

给定一个二叉树，它的每个结点都存放一个 `0-9` 的数字，每条从根到叶子节点的路径都代表一个数字。

例如，从根到叶子节点路径 `1->2->3` 代表数字 `123`。

计算从根到叶子节点生成的所有数字之和。

**说明:** 叶子节点是指没有子节点的节点。

> **示例 1:**

```
输入: [1,2,3]
    1
   / \
  2   3
输出: 25
解释:
从根到叶子节点路径 1->2 代表数字 12.
从根到叶子节点路径 1->3 代表数字 13.
因此，数字总和 = 12 + 13 = 25.
```

> **示例 2:**

```
输入: [4,9,0,5,1]
    4
   / \
  9   0
 / \
5   1
输出: 1026
解释:
从根到叶子节点路径 4->9->5 代表数字 495.
从根到叶子节点路径 4->9->1 代表数字 491.
从根到叶子节点路径 4->0 代表数字 40.
因此，数字总和 = 495 + 491 + 40 = 1026.
```

#### 题目链接

<https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/>

#### **思路:**

　　dfs。用一个全局变量记录`从根结点到当前结点的路径`，当到达叶子结点时结果累加上这条路径的`数值`。  　　

#### **代码:**

```python
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        curr = ''  # 路径字符串
        ans = 0  # 累加结果
        def dfs(node):
            if not node:
                return 

            nonlocal curr, ans
            curr += str(node.val)

            if not node.left and not node.right:
                ans += int(curr)
                curr = curr[:-1]
                return 

            dfs(node.left)
            dfs(node.right)

            curr = curr[:-1]  # 恢复到函数调用前的状态

        dfs(root)
        return ans
      
```

## A130. 被围绕的区域

难度`中等`

#### 题目描述

给定一个二维的矩阵，包含 `'X'` 和 `'O'`（**字母 O**）。

找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。

> **示例:**

```
X X X X
X O O X
X X O X
X O X X
```

运行你的函数后，矩阵变为：

```
X X X X
X X X X
X X X X
X O X X
```

**解释:**

被围绕的区间不会存在于边界上，换句话说，任何边界上的 `'O'` 都不会被填充为 `'X'`。 任何不在边界上，或不与边界上的 `'O'` 相连的 `'O'` 最终都会被填充为 `'X'`。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。

#### 题目链接

<https://leetcode-cn.com/problems/surrounded-regions/>

#### **思路:**

　　先沿着**边界的**每个`"O"`进行dfs，把所有搜索到的`"O"`都替换成`"F"`。  

　　然后把剩下的`”O“`都替换成`"X"`。  

　　最后把`"F"`再替换回`"O"`。

#### **代码:**

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        arounds = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        m = len(board)
        if not m: return 
        n = len(board[0])
        def dfs(i, j):
            if i < 0 or j < 0 or i >= m or j >= n:
                return

            if board[i][j] != 'O':
                return

            board[i][j] = 'F'  # fixed
            for di, dj in arounds:
                dfs(i + di, j + dj)

        for i in range(m):
            for j in range(n):
                if i == 0 or i == m-1 or j == 0 or j == n-1:  # 边界
                    dfs(i, j)
            
        def replace(a, b):
            for i in range(m):
                for j in range(n):
                    if board[i][j] == a:
                        board[i][j] = b

        replace('O', 'X')
        replace('F', 'O')
        
```

## A133. 克隆图

难度`中等`

#### 题目描述

给你无向 **连通** 图中一个节点的引用，请你返回该图的 [**深拷贝**](https://baike.baidu.com/item/深拷贝/22785317?fr=aladdin)（克隆）。

图中的每个节点都包含它的值 `val`（`int`） 和其邻居的列表（`list[Node]`）。

```
class Node {
    public int val;
    public List<Node> neighbors;
} 
```

**测试用例格式：**

简单起见，每个节点的值都和它的索引相同。例如，第一个节点值为 1（`val = 1`），第二个节点值为 2（`val = 2`），以此类推。该图在测试用例中使用邻接列表表示。

**邻接列表** 是用于表示有限图的无序列表的集合。每个列表都描述了图中节点的邻居集。

给定节点将始终是图中的第一个节点（值为 1）。你必须将 **给定节点的拷贝** 作为对克隆图的引用返回。

> **示例 1：**

<img src="_img/133_1.png" style="zoom:25%"/>

```
输入：adjList = [[2,4],[1,3],[2,4],[1,3]]
输出：[[2,4],[1,3],[2,4],[1,3]]
解释：
图中有 4 个节点。
节点 1 的值是 1，它有两个邻居：节点 2 和 4 。
节点 2 的值是 2，它有两个邻居：节点 1 和 3 。
节点 3 的值是 3，它有两个邻居：节点 2 和 4 。
节点 4 的值是 4，它有两个邻居：节点 1 和 3 。
```

> **示例 2：**

<img src="_img/133_2.png" style="zoom:90%"/>

```
输入：adjList = [[]]
输出：[[]]
解释：输入包含一个空列表。该图仅仅只有一个值为 1 的节点，它没有任何邻居。
```

> **示例 3：**

```
输入：adjList = []
输出：[]
解释：这个图是空的，它不含任何节点。
```

> **示例 4：**

<img src="_img/133_4.png" style="zoom:90%"/>

```
输入：adjList = [[2],[1]]
输出：[[2],[1]]
```

#### 题目链接

<https://leetcode-cn.com/problems/clone-graph/>

#### **思路:**

　　用一个字典记录`结点值`和`结点指针`的映射关系，这样在一条**新的边**连接到**旧的结点**上时就能找到之前创建过的结点。  

　　顺便提一下**浅拷贝**和**深拷贝**的区别：**浅拷贝**将原来的图拷贝一遍，增加或删除结点不会影响原来的图，但是浅拷贝**图中的结点还是原来的结点**，修改图中结点的值会影响原图中结点的值。而**深拷贝**将所有的结点都重新初始化一遍，也就是新的图和旧的图完全没有关系了。  

```python
G = [node1, node2, node3, node4]
G_浅拷贝 = [node1, node2, node3, node4]
# G_浅拷贝.append() 或者 G_浅拷贝.pop() 不会影响G
# 但是修改原有node的值(如G[0].val = 1)会影响原来的G
G_深拷贝 = [new_node1, new_node2, new_node3, new_node4]
# G_深拷贝 和 G 已经完全没有关系了
```

#### **代码:**

```python
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        visited = {}
        def dfs(node):
            if not node:
                return None

            temp = Node(node.val)
            visited[node.val] = temp
            for n in node.neighbors:
                if n.val in visited:
                    temp.neighbors.append(visited[n.val])
                    continue
                temp.neighbors.append(dfs(n))      

            return temp

        return dfs(node)

```

## A199. 二叉树的右视图

难度`中等`

#### 题目描述

给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

> **示例:**

```
输入: [1,2,3,null,5,null,4]
输出: [1, 3, 4]
解释:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
```

#### 题目链接

<https://leetcode-cn.com/problems/binary-tree-right-side-view/>

#### **思路:**

　　使用层序遍历，并只保留每层最后一个节点的值。  

#### **代码:**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        
        if not root:
            return []

        queue = [root]
        ans = []
        while queue:
            temp = []
            ans.append(queue[-1].val)
            for q in queue:
                if q.left:
                    temp.append(q.left)
                if q.right:
                    temp.append(q.right)
            
            queue = temp

        return ans

```

## A200. 岛屿数量

难度`中等`

#### 题目描述

给定一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。

> **示例 1:**

```
输入:
11110
11010
11000
00000

输出: 1
```

> **示例 2:**

```
输入:
11000
11000
00100
00011

输出: 3
```

#### 题目链接

<https://leetcode-cn.com/problems/number-of-islands/>

#### **思路:**

　　经典dfs。遍历整个矩阵，从`任意"1"`的位置开始dfs，同时计数`+1`，搜索岛屿的过程中将搜索过的位置都置为`"0"`。最终计数的结果就是岛屿的数量。  

#### **代码:**

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        arounds = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        m = len(grid)
        if not m:
            return 0
        n = len(grid[0])

        ans = 0
        def dfs(i, j):
            if i < 0 or j < 0 or i >= m or j >= n:
                return
            if grid[i][j] == "0":
                return 

            grid[i][j] = "0"
            for di, dj in arounds:
                dfs(i + di, j + dj)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    ans += 1
                    dfs(i, j)

        return ans

```

## A207. 课程表

难度`中等`

#### 题目描述

你这个学期必须选修 `numCourse` 门课程，记为 `0` 到 `numCourse-1` 。

在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们：`[0,1]`

给定课程总量以及它们的先决条件，请你判断是否可能完成所有课程的学习？

> **示例 1:**

```
输入: 2, [[1,0]] 
输出: true
解释: 总共有 2 门课程。学习课程 1 之前，你需要完成课程 0。所以这是可能的。
```

> **示例 2:**

```
输入: 2, [[1,0],[0,1]]
输出: false
解释: 总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0；并且学习课程 0 之前，你还应先完成课程 1。这是不可能的。
```

**提示：**

1. 输入的先决条件是由 **边缘列表** 表示的图形，而不是 邻接矩阵 。详情请参见[图的表示法](http://blog.csdn.net/woaidapaopao/article/details/51732947)。
2. 你可以假定输入的先决条件中没有重复的边。
3. `1 <= numCourses <= 10^5`

#### 题目链接

<https://leetcode-cn.com/problems/course-schedule/>

#### **思路:**

　　拓扑排序。构建的邻接表就是我们通常认识的邻接表，每一个结点存放的是后继结点的集合。

　　该方法的每一步总是输出当前无前趋（即入度为零）的顶点。

　　对应到本题，每一步总是学习**当前无先修课程的**课程。然后把这些学过的课程从其他课程的先修课程中移除。同时把`未学习课程集合`中减去已学习的课程。    

　　最终判断`未学习课程集合`是否为空。  

#### **代码:**

```python
from collections import defaultdict

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool: 
        dict_p = defaultdict(list)
        dict_after = defaultdict(list)

        for curr, pre in prerequisites:
            dict_p[curr].append(pre)  # 邻接表
            dict_after[pre].append(curr)  # 逆邻接表

        # print(dict_p)
        not_learned = set(range(numCourses))  # 未学习课程的集合
        while True:
            new_learned = set()
            for i in not_learned:
                if not dict_p[i]:  # 没有先修课程的课程，都可以学
                    new_learned.add(i)

            if not new_learned:  # 无法学习新课程，结束循环
                break
            for learned in new_learned:
                for after in dict_after[learned]:
                    dict_p[after].remove(learned)  # 从其他课程的先决条件里去掉已经学过的课 

            not_learned = not_learned - new_learned  # 集合差集

        return len(not_learned) == 0
```

## A210. 课程表 II

难度`中等`

#### 题目描述

现在你总共有 *n* 门课需要选，记为 `0` 到 `n-1`。

在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: `[0,1]`

给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。

可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。

> **示例 1:**

```
输入: 2, [[1,0]] 
输出: [0,1]
解释: 总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。
```

> **示例 2:**

```
输入: 4, [[1,0],[2,0],[3,1],[3,2]]
输出: [0,1,2,3] or [0,2,1,3]
解释: 总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。
     因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3] 。
```

**说明:**

1. 输入的先决条件是由**边缘列表**表示的图形，而不是邻接矩阵。详情请参见[图的表示法](http://blog.csdn.net/woaidapaopao/article/details/51732947)。
2. 你可以假定输入的先决条件中没有重复的边。

**提示:**

1. 这个问题相当于查找一个循环是否存在于有向图中。如果存在循环，则不存在拓扑排序，因此不可能选取所有课程进行学习。
2. [通过 DFS 进行拓扑排序](https://www.coursera.org/specializations/algorithms) - 一个关于Coursera的精彩视频教程（21分钟），介绍拓扑排序的基本概念。
3. 拓扑排序也可以通过 [BFS](https://baike.baidu.com/item/宽度优先搜索/5224802?fr=aladdin&fromid=2148012&fromtitle=广度优先搜索) 完成。

#### 题目链接

<https://leetcode-cn.com/problems/course-schedule-ii/>

#### **思路:**

　　和上一题[A207. 课程表](/dfs?id=a207-课程表)一样，新增记录顺序即可。  

#### **代码:**

```python
from collections import defaultdict

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        dict_p = defaultdict(list)
        dict_after = defaultdict(list)

        for curr, pre in prerequisites:
            dict_p[curr].append(pre)  # 邻接表
            dict_after[pre].append(curr)  # 逆邻接表

        # print(dict_p)
        not_learned = set(range(numCourses))  # 未学习课程的集合
        ans = []
        while True:
            new_learned = set()
            for i in not_learned:
                if not dict_p[i]:  # 没有先修课程的课程，都可以学
                    new_learned.add(i)

            if not new_learned:  # 无法学习新课程，结束循环
                break
            for learned in new_learned:
                ans.append(learned)
                for after in dict_after[learned]:
                    dict_p[after].remove(learned)  # 从其他课程的先决条件里去掉已经学过的课 

            not_learned = not_learned - new_learned  # 集合差集

        if len(not_learned) == 0:  # 能学完所有课程
            return ans
        else:
            return []
```

## A257. 二叉树的所有路径

难度`简单`

#### 题目描述

给定一个二叉树，返回所有从根节点到叶子节点的路径。

**说明:** 叶子节点是指没有子节点的节点。

> **示例:**

```
输入:

   1
 /   \
2     3
 \
  5

输出: ["1->2->5", "1->3"]

解释: 所有根节点到叶子节点的路径为: 1->2->5, 1->3
```

#### 题目链接

<https://leetcode-cn.com/problems/binary-tree-paths/>

#### **思路:**

　　dfs。用一个全局变量`curr`记录`从根结点到当前结点的路径`，当到达叶子结点时记录这条路径。  

#### **代码:**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        curr = []  # 路径列表
        ans = []  # 结果结果
        def dfs(node):
            if not node:
                return 

            curr.append(str(node.val))
            idx = len(curr) - 1  # 记录插入的位置，方便函数返回前弹出

            if not node.left and not node.right:  # 到达叶子结点
                ans.append('->'.join(curr))
                curr.pop(idx)
                return 

            dfs(node.left)
            dfs(node.right)

            curr.pop(idx)  # 恢复到函数调用前的状态

        dfs(root)
        return ans
```

## A301. 删除无效的括号

难度`困难`

#### 题目描述

删除最小数量的无效括号，使得输入的字符串有效，返回所有可能的结果。

**说明:** 输入可能包含了除 `(` 和 `)` 以外的字符。

> **示例 1:**

```
输入: "()())()"
输出: ["()()()", "(())()"]
```

> **示例 2:**

```
输入: "(a)())()"
输出: ["(a)()()", "(a())()"]
```

> **示例 3:**

```
输入: ")("
输出: [""]
```

#### 题目链接

<https://leetcode-cn.com/problems/remove-invalid-parentheses/>

#### **思路:**

　　dfs。第一遍搜索找到`最少删除括号的个数`。  

　　第二遍搜索以`最少删除括号的个数`为剪枝条件，寻找所有括号匹配的可能性。  

　　用`set`去除重复。

#### **代码:**

```python
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        ls = len(s)
        ans = set()
        minimum_delete = ls  # 最小删除的括号数
        def dfs(i, cnt, del_cnt, max_delte, cur): # s[i:] cnt个左括号 删了几个 最多删几个括号, 当前字符串
            nonlocal minimum_delete
            if del_cnt > max_delte:
                return False
            if i >= ls:
                if cnt == 0:
                    minimum_delete = del_cnt
                    # print(del_cnt)
                    ans.add(cur)
                    if max_delte == float('inf'):
                        return True  # return True可以确保找最小删除数的时候不重复搜索
                return False

            if s[i] == '(':  # 要么用这个左括号 要么不用
                if dfs(i+1, cnt+1, del_cnt, max_delte, cur + '('):  # 用(
                    return True
                return dfs(i+1, cnt, del_cnt+1, max_delte, cur) # 不用(
              
            elif s[i] == ')':
                if cnt > 0 and dfs(i+1, cnt - 1, del_cnt, max_delte, cur+')'):  # 用)
                    return True
                return dfs(i+1, cnt, del_cnt+1, max_delte, cur)   # 不用)
              
            else:  # 非括号字符
                return dfs(i+1, cnt, del_cnt, max_delte, cur + s[i])
            
        
        dfs(0, 0, 0, float('inf'), '')  # 第一次dfs，找到最少删几个括号
        ans.clear()
        dfs(0, 0, 0, minimum_delete, '')  # 第二次dfs，找到所有的结果
        return [a for a in ans]
      
```

## A329. 矩阵中的最长递增路径

难度`困难`

#### 题目描述

难度困难140收藏分享切换为英文关注反馈

给定一个整数矩阵，找出最长递增路径的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 你不能在对角线方向上移动或移动到边界外（即不允许环绕）。

> **示例 1:**

```
输入: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
输出: 4 
解释: 最长递增路径为 [1, 2, 6, 9]。
```

> **示例 2:**

```
输入: nums = 
[
  [3,4,5],
  [3,2,6],
  [2,2,1]
] 
输出: 4 
解释: 最长递增路径是 [3, 4, 5, 6]。注意不允许在对角线方向上移动。
```

#### 题目链接

<https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/>

#### **思路:**

　　**方法一：**dfs记忆化搜索。先从矩阵中找到所有**四周元素都不比自己小**的元素作为起点，从它们开始dfs。用数组`dp[i][j]`记录从开始位置到某个位置的最长路径，如果某个元素不能使周围元素的`dp`变大，就不再继续往下搜索了，起到剪枝的效果。  　　

　　**方法二：**动态规划。先预处理，对矩阵的值按从小到大排序，按大小顺序才能保证依赖的子问题都求解过了。

　　`dp[i][j]`表示以`matrix[i][j]`结尾的最长递增长度。

- 初始`dp[i][j]`都等于1；  
- 若`matrix[i][j]`四个方向有任意小于它，则可以更新`dp[i][j] = max(dp[i][j], 1 + dp[r][c])`。  

#### **代码:**  

　　**方法一：**dfs记忆化搜索 (1036 ms)  

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        arounds = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        # matrix = grid = board
        m = len(matrix)
        if not m: return 0
        n = len(matrix[0])
        dp = [[1 for _ in range(n)] for _ in range(m)]
        visited = [[False for _ in range(n)] for _ in range(m)]
        ans = 0

        def dfs(i, j, depth, from_num):
            nonlocal ans
            if i < 0 or j < 0 or i >= m or j >= n:  # 边界
                return

            if visited[i][j] or matrix[i][j] <= from_num:  # 不能走
                return

            visited[i][j] = True
            depth += 1
            ans = max(ans, depth)
            dp[i][j] = depth

            temp = []
            for di, dj in arounds:
                x, y = i + di, j+ dj
                if x < 0 or y < 0 or x >= m or y >= n or visited[x][y]:
                    continue
                if dp[x][y] < depth + 1:  # 无法更优就不搜索了
                    temp.append((matrix[x][y] - matrix[i][j], x, y))

            temp.sort()  # 从相邻的数字中小的开始搜索
            for _, x, y in temp:
                dfs(x, y, depth, matrix[i][j])

            visited[i][j] = False

        def get(i, j):
            if i < 0 or j < 0 or i >= m or j >= n:  # 边界
                return float('inf')
            return matrix[i][j]

        for i in range(m):
            for j in range(n):
                num = matrix[i][j]
                if all([get(i + di, j + dj) >= num for di, dj in arounds]):  # 四周没有更小的数
                    dfs(i, j, 0, float('-inf'))

        return ans
```

　　**方法二：**动态规划 (516ms)

```python
class Solution(object):
    def longestIncreasingPath(self, matrix):
        if not matrix or not matrix[0]:
            return 0
        m, n = len(matrix), len(matrix[0])
        lst = []
        for i in range(m):
            for j in range(n):
                lst.append((matrix[i][j], i, j))
        lst.sort()
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for num, i, j in lst:
            dp[i][j] = 1
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                r, c = i + di, j + dj
                if 0 <= r < m and 0 <= c < n:
                    if matrix[i][j] > matrix[r][c]:
                        dp[i][j] = max(dp[i][j], 1 + dp[r][c])
        return max([dp[i][j] for i in range(m) for j in range(n)])
      
```

## A332. 重新安排行程

难度`中等`

#### 题目描述

给定一个机票的字符串二维数组 `[from, to]`，子数组中的两个成员分别表示飞机出发和降落的机场地点，对该行程进行重新规划排序。所有这些机票都属于一个从JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 出发。

**说明:**

1. 如果存在多种有效的行程，你可以按字符自然排序返回最小的行程组合。例如，行程 ["JFK", "LGA"] 与 ["JFK", "LGB"] 相比就更小，排序更靠前
2. 所有的机场都用三个大写字母表示（机场代码）。
3. 假定所有机票至少存在一种合理的行程。

> **示例 1:**

```
输入: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
输出: ["JFK", "MUC", "LHR", "SFO", "SJC"]
```

> **示例 2:**

```
输入: [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
输出: ["JFK","ATL","JFK","SFO","ATL","SFO"]
解释: 另一种有效的行程是 ["JFK","SFO","ATL","JFK","ATL","SFO"]。但是它自然排序更大更靠后。
```

#### 题目链接

<https://leetcode-cn.com/problems/reconstruct-itinerary/>

#### **思路:**

　　求欧拉路径(一笔画问题)的栈版本，每次入栈的是字母序最小的。如果栈顶的结点没有相邻的结点就出栈。 

　　将出栈的次序倒序排列就是最终的结果。   

<img src="_img/a332.png" style="zoom:60%"/>

#### **代码:**

```python
from collections import defaultdict

class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        l = len(tickets)
        dict_t = defaultdict(list)
        for f, t in tickets:
            dict_t[f].append(t)

        stack = ['JFK']  # 初始位置
        ans = []
        while stack:
            curr = stack[-1]  # peek
            if dict_t[curr]:
                next = min(dict_t[curr])
                dict_t[curr].remove(next)
                stack.append(next)
            else:
                ans.append(curr)
                stack.pop()

        return ans[::-1]
      
```

## A337. 打家劫舍 III

难度`中等`

#### 题目描述

在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

> **示例 1:**

```
输入: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

输出: 7 
解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
```

> **示例 2:**

```
输入: [3,4,5,1,3,null,1]

     3
    / \
   4   5
  / \   \ 
 1   3   1

输出: 9
解释: 小偷一晚能够盗取的最高金额 = 4 + 5 = 9.
```

#### 题目链接

<https://leetcode-cn.com/problems/house-robber-iii/>

#### **思路:**

　　对于任意一个结点`node`，只有两种状态，要么偷，要么不偷，分别计算这这种情况的最大值即可。  

<img src="_img/a337.png" style="zoom:50%"/>

　　如果偷结点`node`，就不能偷`node`的子结点，最大值为`not_rob_left`+`not_rob_right`+`node.val`。  

　　如果不偷结点`node`，可以偷`node`的子结点(也可以不偷)，最大值为max(`rob_left`,`not_rob_left`) + max(`rob_right`,`not_rob_right`)。  

#### **代码:**

```python
class Solution:
    def rob(self, root: TreeNode) -> int:
        def dfs(node):  # 返回use not_use
            if not node:
                return 0, 0

            if not node.left and not node.right:  # 叶子结点
                return node.val, 0

            rob_left, no_rob_left = dfs(node.left)
            rob_right, no_rob_right = dfs(node.right)
            
            return no_rob_left + no_rob_right + node.val, max(rob_left, no_rob_left) + max(rob_right, no_rob_right)

        rob_root, no_rob_root = dfs(root)
        return max(rob_root, no_rob_root)
      
```





