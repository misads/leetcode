# BFS(广度优先搜索)

## A102. 二叉树的层序遍历

难度`中等`

#### 题目描述

给你一个二叉树，请你返回其按 **层序遍历** 得到的节点值。 （即逐层地，从左到右访问所有节点）。

**示例：**
二叉树：`[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回其层次遍历结果：

```
[
  [3],
  [9,20],
  [15,7]
]
```

#### 题目链接

<https://leetcode-cn.com/problems/binary-tree-level-order-traversal/>

#### **思路:**

　　[层序遍历模板](/实用模板?id=广搜：bfs🌲层序遍历)。  

#### **代码:**

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        queue = [root]
        ans = []
        while queue:
            ans.append([q.val for q in queue])
            temp = []
            # queue存放的是当前层的所有结点
            for q in queue:
                if q.left:
                    temp.append(q.left)
                if q.right:
                    temp.append(q.right)

            queue = temp
        return ans

```

## A103. 二叉树的锯齿形层次遍历

难度`中等`

#### 题目描述

给定一个二叉树，返回其节点值的锯齿形层次遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

例如：
给定二叉树 `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回锯齿形层次遍历如下：

```
[
  [3],
  [20,9],
  [15,7]
]
```

#### 题目链接

<https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/>

#### **思路:**

　　和上一题一样。用一个`flag`标记是从左往右还是从右往左就行了。  

#### **代码:**

```python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        queue = [root]
        ans = []
        left_to_right = True
        while queue:
            if left_to_right:
                ans.append([q.val for q in queue])
            else:
                ans.append([q.val for q in queue[::-1]])
            left_to_right = not left_to_right
            temp = []
            # queue存放的是当前层的所有结点
            for q in queue:
                if q.left:
                    temp.append(q.left)
                if q.right:
                    temp.append(q.right)

            queue = temp
        return ans

```

## A127. 单词接龙

难度`中等`

#### 题目描述

给定两个单词（*beginWord* 和 *endWord* ）和一个字典，找到从 *beginWord* 到 *endWord* 的最短转换序列的长度。转换需遵循如下规则：

1. 每次转换只能改变一个字母。
2. 转换过程中的中间单词必须是字典中的单词。

**说明:**

- 如果不存在这样的转换序列，返回 0。
- 所有单词具有相同的长度。
- 所有单词只由小写字母组成。
- 字典中不存在重复的单词。
- 你可以假设 *beginWord* 和 *endWord* 是非空的，且二者不相同。

> **示例 1:**

```
输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

输出: 5

解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",
     返回它的长度 5。
```

> **示例 2:**

```
输入:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

输出: 0

解释: endWord "cog" 不在字典中，所以无法进行转换。
```

#### 题目链接

<https://leetcode-cn.com/problems/word-ladder/)>

#### **思路:**

　　标准的bfs。这题主要的时间花在**找相邻的结点上**。  

　　如果每次都遍历`wordList`去找相邻的结点，要花费大量的时间(时间复杂度`O(n^2)`)。因此采用把单词的某个字母替换成其他小写字母的方式，时间复杂度仅为`O(n*l)`，`l`为单词长度。  

#### **代码:**

```python
from collections import defaultdict

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0

        d = defaultdict(list)
        for word in wordList:
            for i in range(len(word)):
                d[word[:i] + "*" + word[i + 1:]].append(word)

        visited = {i: False for i in wordList}
        visited[beginWord] = False

        queue = [beginWord]
        depth = 1
        while queue:
            temp = []
            for q in queue:
                visited[q] = True

            for q in queue:
                if q == endWord:
                    return depth  # 到达终点

                for i in range(len(q)):
                    key = q[:i] + "*" + q[i + 1:]
                    for neibour in d[key]:
                        if not visited[neibour]:
                            temp.append(neibour)
            depth += 1
            queue = temp
            del temp

        return 0

```

## A310. 最小高度树

难度`中等`

#### 题目描述

对于一个具有树特征的无向图，我们可选择任何一个节点作为根。图因此可以成为树，在所有可能的树中，具有最小高度的树被称为最小高度树。给出这样的一个图，写出一个函数找到所有的最小高度树并返回他们的根节点。

**格式**

该图包含 `n` 个节点，标记为 `0` 到 `n - 1`。给定数字 `n` 和一个无向边 `edges` 列表（每一个边都是一对标签）。

你可以假设没有重复的边会出现在 `edges` 中。由于所有的边都是无向边， `[0, 1]`和 `[1, 0]` 是相同的，因此不会同时出现在 `edges` 里。

> **示例 1:**

```
输入: n = 4, edges = [[1, 0], [1, 2], [1, 3]]

        0
        |
        1
       / \
      2   3 

输出: [1]
```

> **示例 2:**

```
输入: n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]

     0  1  2
      \ | /
        3
        |
        4
        |
        5 

输出: [3, 4]
```

#### 题目链接

<https://leetcode-cn.com/problems/minimum-height-trees/>

#### **思路:**

　　构建图，循环遍历图，找出叶子节点。去除叶子节点。直到图中节点只剩下2个或1个。返回剩下的节点。  

#### **代码:**

```python
from collections import defaultdict

class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if not n:
            return []
        if not edges:
            return list(range(n))

        graph = defaultdict(list)
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)

        nodes = set(range(n))
        while True:
            if len(nodes) <= 2:
                return [n for n in nodes]
                
            leaves = []
            for k in graph:
                if len(graph[k]) == 1:
                    leaves.append(k)
                    nodes.remove(k)

            for k in leaves:
                    o = graph[k][0]
                    graph[k].clear()
                    graph[o].remove(k)

```

## A417. 太平洋大西洋水流问题

难度`中等`

#### 题目描述

给定一个 `m x n` 的非负整数矩阵来表示一片大陆上各个单元格的高度。“太平洋”处于大陆的左边界和上边界，而“大西洋”处于大陆的右边界和下边界。

规定水流只能按照上、下、左、右四个方向流动，且只能从高到低或者在同等高度上流动。

请找出那些水流既可以流动到“太平洋”，又能流动到“大西洋”的陆地单元的坐标。 

**提示：**

1. 输出坐标的顺序不重要
2. *m* 和 *n* 都小于150

> **示例：** 

```
给定下面的 5x5 矩阵:

  太平洋 ~   ~   ~   ~   ~ 
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * 大西洋

返回:

[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (上图中带括号的单元).
```

#### 题目链接

<https://leetcode-cn.com/problems/pacific-atlantic-water-flow/>

#### **思路:**

　　类似泛洪的思想，从太平洋和大西洋逆流往上，分别bfs，记录所有能到达的点，然后取交集。  

#### **代码:**

```python
class Solution:
    def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
        m = len(matrix)
        if not m:
            return []
        n = len(matrix[0])

        pacific = []
        atlantic = []
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    pacific.append((i, j))
                if i == m - 1 or j == n - 1:
                    atlantic.append((i, j))

        arounds = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def bfs(queue):
            can_reach = set()
            visited = [[False for _ in range(n)] for _ in range(m)]
            while queue:
                for i, j in queue:
                    visited[i][j] = True
                    can_reach.add((i, j))

                temp = []
                for i, j in queue:
                    for di, dj in arounds:
                        x, y = i + di, j + dj
                        if x < 0 or y < 0 or x >= m or y >= n:
                            continue
                        if not visited[x][y] and matrix[x][y] >= matrix[i][j]:
                            temp.append((x, y))

                queue = temp
            return can_reach

        a = bfs(pacific)
        b = bfs(atlantic)
        c = a & b
        return [_ for _ in c]
```


## A429. N叉树的层序遍历

难度`中等`

#### 题目描述

给定一个 N 叉树，返回其节点值的*层序遍历*。 (即从左到右，逐层遍历)。

例如，给定一个 `3叉树` :

<img src="_img/429.png" style="zoom:40%"/>

 

返回其层序遍历:

```
[
     [1],
     [3,2,4],
     [5,6]
]
```

**说明:**

1. 树的深度不会超过 `1000`。
2. 树的节点总数不会超过 `5000`。

#### 题目链接

<https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/>

#### **思路:**

　　[层序遍历模板](/实用模板?id=广搜：bfs🌲层序遍历)。  

#### **代码:**

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:
            return 

        queue = [root]
        ans = []
        while queue:
            temp = []
            ans.append([q.val for q in queue])
            # queue存放的是当前层的所有结点
            for q in queue:
                for children in q.children:
                    temp.append(children)

            queue = temp
        return ans
      
```

## A513. 找树左下角的值

难度`中等`

#### 题目描述

给定一个二叉树，在树的最后一行找到最左边的值。

> **示例 1:**

```
输入:

    2
   / \
  1   3

输出:
1 
```

> **示例 2:**

```
输入:

        1
       / \
      2   3
     /   / \
    4   5   6
       /
      7

输出:
7
```

#### 题目链接

<https://leetcode-cn.com/problems/find-bottom-left-tree-value/>

#### **思路:**

　　[层序遍历模板](/实用模板?id=广搜：bfs🌲层序遍历)。 

#### **代码:**

```python
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        if not root:
            return 

        queue = [root]
        ans = 0
        while queue:
            temp = []
            # queue存放的是当前层的所有结点
            for q in queue:
                if q.left:
                    temp.append(q.left)
                if q.right:
                    temp.append(q.right)
            if not temp:
                ans = queue[0].val

            queue = temp
        return ans
```

## A515. 在每个树行中找最大值

难度`中等`

#### 题目描述

您需要在二叉树的每一行中找到最大的值。

> **示例：**

```
输入: 

          1
         / \
        3   2
       / \   \  
      5   3   9 

输出: [1, 3, 9]
```

#### 题目链接

<https://leetcode-cn.com/problems/find-largest-value-in-each-tree-row/>

#### **思路:**

　　[层序遍历模板](/实用模板?id=广搜：bfs🌲层序遍历)。 

#### **代码:**

```python

class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        if not root:
            return 

        queue = [root]
        ans = []
        while queue:
            temp = []
            ans.append(max([node.val for node in queue]))
            # queue存放的是当前层的所有结点
            for q in queue:
                if q.left:
                    temp.append(q.left)
                if q.right:
                    temp.append(q.right)

            queue = temp
        return ans
```



## A529. 扫雷游戏

难度`中等`

#### 题目描述

让我们一起来玩扫雷游戏！

给定一个代表游戏板的二维字符矩阵。 **'M'** 代表一个**未挖出的**地雷，**'E'** 代表一个**未挖出的**空方块，**'B'** 代表没有相邻（上，下，左，右，和所有4个对角线）地雷的**已挖出的**空白方块，**数字**（'1' 到 '8'）表示有多少地雷与这块**已挖出的**方块相邻，**'X'** 则表示一个**已挖出的**地雷。

现在给出在所有**未挖出的**方块中（'M'或者'E'）的下一个点击位置（行和列索引），根据以下规则，返回相应位置被点击后对应的面板：

1. 如果一个地雷（'M'）被挖出，游戏就结束了- 把它改为 **'X'**。
2. 如果一个**没有相邻地雷**的空方块（'E'）被挖出，修改它为（'B'），并且所有和其相邻的方块都应该被递归地揭露。
3. 如果一个**至少与一个地雷相邻**的空方块（'E'）被挖出，修改它为数字（'1'到'8'），表示相邻地雷的数量。
4. 如果在此次点击中，若无更多方块可被揭露，则返回面板。

> **示例 1：**

```
输入: 

[['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'M', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E']]

Click : [3,0]

输出: 

[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'M', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

解释:
```

<img src="_img/529_1.png" style="zoom:40%"/>  

> **示例 2：**

```
输入: 

[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'M', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

Click : [1,2]

输出: 

[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'X', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

解释:
```

<img src="_img/529_2.png" style="zoom:40%"/>

**注意：**

1. 输入矩阵的宽和高的范围为 [1,50]。
2. 点击的位置只能是未被挖出的方块 ('M' 或者 'E')，这也意味着面板至少包含一个可点击的方块。
3. 输入面板不会是游戏结束的状态（即有地雷已被挖出）。
4. 简单起见，未提及的规则在这个问题中可被忽略。例如，当游戏结束时你不需要挖出所有地雷，考虑所有你可能赢得游戏或标记方块的情况。

#### 题目链接

<https://leetcode-cn.com/problems/minesweeper/>

#### **思路:**

　　本题的难点在于理解题意。根据点击的位置不同可以分为以下三种情况：    

- 点击的位置是地雷，将该位置修改为`"X"`，直接返回。  
- 点击位置的`九宫格内有地雷`，将该位置修改为九宫格内地雷的数量，然后直接返回。
- 点击位置的`九宫格内没有地雷`，进行dfs或bfs向周围扩展，直到所有边界的九宫格内都有地雷。  

#### **代码:**

```python
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        x, y = click
        if board[x][y] == 'M':  # 踩地雷了
            board[x][y] = 'X'
            return board

        m = len(board)
        n = len(board[0])
        counts = [[0 for _ in range(n)] for _ in range(m)]
        visited = [[False for _ in range(n)] for _ in range(m)]
        arounds = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8个方向

        def get_around(i, j):  # 获取周围九宫格
            count = 0
            for x in range(max(0, i-1), min(m, i+2)):
                for y in range(max(0, j-1), min(n, j+2)):
                    if (x != i or y != j) and board[x][y] == 'M':
                        count += 1
            return count

        for i in range(m):
            for j in range(n):
                counts[i][j] = get_around(i, j)  # 每个位置的数字

        if counts[x][y]:  # 如果点的位置是有数字的
            board[x][y] = str(counts[x][y])
            return board

        queue = [(x, y)]  # bfs
        while queue:
            for i, j in queue:
                visited[i][j] = True
                if board[i][j] != 'M':  # 地雷不能修改
                    board[i][j] = str(counts[i][j]) if counts[i][j] else 'B'

            temp = []
            for i, j in queue:
                if counts[i][j]:  # 不是0的位置不继续
                    continue
                for di, dj in arounds:
                    ni, nj = i + di, j + dj  # 下一个位置
                    if ni < 0 or nj < 0 or ni >= m or nj >= n:  # 边界
                        continue
                    if not visited[ni][nj] and (ni, nj) not in temp:
                        temp.append((ni, nj))

            queue = temp
        return board
      
```

## A542. 01 矩阵

难度`中等`

#### 题目描述

给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。

两个相邻元素间的距离为 1 。

> **示例 1:**  

输入:

```
0 0 0
0 1 0
0 0 0
```

输出:

```
0 0 0
0 1 0
0 0 0
```

> **示例 2:** 

输入:

```
0 0 0
0 1 0
1 1 1
```

输出:

```
0 0 0
0 1 0
1 2 1
```

#### 题目链接



#### **思路:**

　　**方法一：**bfs。将所有的`"0"`看做一个整体，向其他数字的位置腐蚀(将它们变成0)，当格子上所有数字都为`"0"`时结束循环。记录循环的深度就是结果。  

　　**方法二：**动态规划，因为最近的`"0"`要么在左上方，要么在右下方。因此只要分**别从左上角到右下角**和**从右下角到左上角**动态规划一次，就得到了最终的结果。  

　　其中从左上到右下的状态转移方程为`dp[i][j] = min(dp[i][j], dp[i-1][j] + 1, dp[i][j-1] + 1)`。  　　

#### **代码:**

　　**方法一：**(bfs)

```python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        arounds = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        grid = matrix
        m = len(matrix)
        if not m: return []
        n = len(matrix[0])

        ans = [[0 for _ in range(n)] for _ in range(m)]
        temp = []

        def rot(x, y):
            if x < 0 or y < 0 or x >= len(grid) or y >= len(grid[0]):
                return False
            if grid[x][y] != 0:  # 成功腐烂新鲜的橘子，返回True
                grid[x][y] = 0  
                temp.append((x, y))
                return True
            return False

        depth = 0
        queue = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:  # 腐烂的橘子
                    queue.append((i, j))

        while queue: 
            temp = []
            for i, j in queue:
                ans[i][j] = depth
                for di, dj in arounds:
                    rot(i + di, j + dj)

            depth += 1

            queue = temp

        return ans
```

　　**方法二：**(dp)

```python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        arounds = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        grid = matrix
        m = len(matrix)
        if not m: return []
        n = len(matrix[0])

        dp = [[float('inf') for _ in range(n)] for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    dp[i][j] = 0

        for i in range(m):  # 左上到右下
            for j in range(n):
                if i > 0:
                    dp[i][j] = min(dp[i][j], dp[i-1][j] + 1)
                if j > 0:
                    dp[i][j] = min(dp[i][j], dp[i][j-1] + 1)

        for i in range(m-1,-1,-1):  # 右下到左上
            for j in range(n-1,-1,-1):
                if i < m-1:
                    dp[i][j] = min(dp[i][j], dp[i+1][j] + 1)
                if j < n-1:
                    dp[i][j] = min(dp[i][j], dp[i][j+1] + 1)
                    
        return dp
      
```

## A675. 为高尔夫比赛砍树

难度`困难`

#### 题目描述

你被请来给一个要举办高尔夫比赛的树林砍树. 树林由一个非负的二维数组表示， 在这个数组中：

1. `0` 表示障碍，无法触碰到.
2. `1` 表示可以行走的地面.
3. `比 1 大的数` 表示一颗允许走过的树的高度.

每一步，你都可以向上、下、左、右四个方向之一移动一个单位，如果你站的地方有一棵树，那么你可以决定是否要砍倒它。

你被要求按照树的高度从低向高砍掉所有的树，每砍过一颗树，树的高度变为 1 。

你将从（0，0）点开始工作，你应该返回你砍完所有树需要走的最小步数。 如果你无法砍完所有的树，返回 -1 。

可以保证的是，没有两棵树的高度是相同的，并且你至少需要砍倒一棵树。

> **示例 1:**

```
输入: 
[
 [1,2,3],
 [0,0,4],
 [7,6,5]
]
输出: 6
```

> **示例 2:**

```
输入: 
[
 [1,2,3],
 [0,0,0],
 [7,6,5]
]
输出: -1
```

> **示例 3:**

```
输入: 
[
 [2,3,4],
 [0,0,5],
 [8,7,6]
]
输出: 6
解释: (0,0) 位置的树，你可以直接砍去，不用算步数
```

**提示：**

- `1 <= forest.length <= 50`
- `1 <= forest[i].length <= 50`
- `0 <= forest[i][j] <= 10^9`

#### 题目链接

<https://leetcode-cn.com/problems/cut-off-trees-for-golf-event/>

#### **思路:**

　　`"0"`的位置是障碍物，其他的位置(**包括有树的位置**)都是可以走的。  

　　先从出发点`(0, 0)`开始，找与最矮的一棵树之间的最短路径(用bfs)，然后砍倒这棵树，再找到剩下树中最矮的一棵的最短路径。最后所有的路径和就是结果。  

　　如果有树被障碍物挡住，返回`-1`。  

#### **代码:**

```python
class Solution:
    def cutOffTree(self, forest: List[List[int]]) -> int:
        arounds = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        m = len(forest)
        if not m: return 0
        n = len(forest[0])

        dp = [[float('inf') for _ in range(n)] for _ in range(m)]
        dp[0][0] = 0
        sorted_trees = []
        for i in range(m):
             for j in range(n):
                if forest[i][j] > 1:
                     sorted_trees.append((forest[i][j], i, j))

        sorted_trees.sort()

        def from_to(start, end_x, end_y):  # bfs最短路径模板
            visited = {start}
            queue = [start]

            count = 0
            while queue:
                for i, j in queue:
                    if i == end_x and j == end_y:
                        return count
                      
                count += 1

                temp = []
                for i, j in queue:
                    for di, dj in arounds:
                        x, y = i + di, j + dj
                        if x < 0 or y < 0 or x >= m or y >= n or forest[x][y] == 0:  # 边界或障碍
                            continue
                        if (x,y) not in visited: 
                            visited.add((x,y))
                            temp.append((x, y))

                queue = temp

            return -1

        ans = 0
        cur = (0, 0)
        for _, x, y in sorted_trees:
            next_tree = from_to(cur, x, y)
            if next_tree == -1:
                return -1
            ans += next_tree
            cur = (x, y)

        return ans

```

## A690. 员工的重要性

难度`简单`

#### 题目描述

给定一个保存员工信息的数据结构，它包含了员工**唯一的id**，**重要度** 和 **直系下属的id**。

比如，员工1是员工2的领导，员工2是员工3的领导。他们相应的重要度为15, 10, 5。那么员工1的数据结构是[1, 15, [2]]，员工2的数据结构是[2, 10, [3]]，员工3的数据结构是[3, 5, []]。注意虽然员工3也是员工1的一个下属，但是由于**并不是直系**下属，因此没有体现在员工1的数据结构中。

现在输入一个公司的所有员工信息，以及单个员工id，返回这个员工和他所有下属的重要度之和。

> **示例 1:**

```
输入: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
输出: 11
解释:
员工1自身的重要度是5，他有两个直系下属2和3，而且2和3的重要度均为3。因此员工1的总重要度是 5 + 3 + 3 = 11。
```

**注意:**

1. 一个员工最多有一个**直系**领导，但是可以有多个**直系**下属
2. 员工数量不超过2000。

#### 题目链接

<https://leetcode-cn.com/problems/employee-importance/>

#### **思路:**

　　递归。  

#### **代码:**

```python
from collections import defaultdict

class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        subs = defaultdict(list)
        importance = defaultdict(int)
        for e in employees:
            subs[e.id] = e.subordinates
            importance[e.id] = e.importance
            
        def dfs(node):
            me = importance[node]
            for sub in subs[node]:
                me += dfs(sub)
            return me

        return dfs(id)
      
```

## A743. 网络延迟时间

难度`中等`

#### 题目描述

有 `N` 个网络节点，标记为 `1` 到 `N`。

给定一个列表 `times`，表示信号经过**有向**边的传递时间。 `times[i] = (u, v, w)`，其中 `u` 是源节点，`v` 是目标节点， `w` 是一个信号从源节点传递到目标节点的时间。

现在，我们从某个节点 `K` 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 `-1`。 

> **示例：**

<img src="_img/743.png" style="zoom:100%"/>

```
输入：times = [[2,1,1],[2,3,1],[3,4,1]], N = 4, K = 2
输出：2 
```

**注意:**

1. `N` 的范围在 `[1, 100]` 之间。
2. `K` 的范围在 `[1, N]` 之间。
3. `times` 的长度在 `[1, 6000]` 之间。
4. 所有的边 `times[i] = (u, v, w)` 都有 `1 <= u, v <= N` 且 `0 <= w <= 100`。

#### 题目链接

<https://leetcode-cn.com/problems/network-delay-time/>

#### **思路:**

　　由于边的权是不同的**正数**，该题是不含负环的**单源最短路径**问题。  

　　Dijkstra算法是用来求单源最短路径问题，即给定图G和起点s，通过算法得到s到达其他每个顶点的最短距离。

　　基本思想：对图`G(V,E)`设置集合`S`，存放已被访问的顶点，然后每次从集合`V-S`中选择与起点s的最短距离最小的一个顶点（记为u），访问并加入集合`S`。之后，令u为中介点，优化起点s与所有从u能够到达的顶点v之间的最短距离。这样的操作执行n次（n为顶点个数），直到集合`S`已经包含所有顶点。　

#### **代码:**

```python
from collections import defaultdict

class Solution:
    def networkDelayTime(self, times: List[List[int]], N: int, K: int) -> int:
        edge = defaultdict(list)  # edge[1] = (t, 2)
        for u, v, w in times:
            edge[u].append((w, v))

        minimal = [float('inf') for _ in range(N + 1)]
        minimal[K] = 0
        S = {K} # 起点
        VS = edge[K]
        for t, node in edge[K]:
            minimal[node] = t

        while VS:
            t, u = min(VS)
            VS.remove((t, u))
            S.add(u)
            for path in edge[u]:
                if path[1] not in S:
                    VS.append(path)  # 防止出现环
            for t, node in edge[u]:
                minimal[node] = min(minimal[node], minimal[u] + t)  # 经过u为中介

        # print(minimal)
        ans = max(minimal[1:])
        return ans if ans != float('inf') else -1

```

## A752. 打开转盘锁

难度`中等`

#### 题目描述

你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： `'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'` 。每个拨轮可以自由旋转：例如把 `'9'` 变为  `'0'`，`'0'` 变为 `'9'` 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 `'0000'` ，一个代表四个拨轮的数字的字符串。

列表 `deadends` 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 `target` 代表可以解锁的数字，你需要给出最小的旋转次数，如果无论如何不能解锁，返回 -1。

> **示例 1:**

```
输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。
```

> **示例 2:**

```
输入: deadends = ["8888"], target = "0009"
输出：1
解释：
把最后一位反向旋转一次即可 "0000" -> "0009"。
```

> **示例 3:**

```
输入: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
输出：-1
解释：
无法旋转到目标数字且不被锁定。
```

> **示例 4:**

```
输入: deadends = ["0000"], target = "8888"
输出：-1
```

**提示：**

1. 死亡列表 `deadends` 的长度范围为 `[1, 500]`。
2. 目标数字 `target` 不会在 `deadends` 之中。
3. 每个 `deadends` 和 `target` 中的字符串的数字会在 10,000 个可能的情况 `'0000'` 到 `'9999'` 中产生。

#### 题目链接

<https://leetcode-cn.com/problems/open-the-lock/>

#### **思路:**

　　最短路径问题，四位数字(正反转)相当于`8个方向`，`deadends`相当于障碍物。套用bfs模板。  

　　需要注意的是相邻数字的处理，由于`Python3`取模的结果一定是正数，也就是`-1 % 10 = 9`，所以可以用取模来简化代码。四位数字中的任意一位`±1`然后`%10`就可以得到下一种可能性。  

#### **代码:**

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        if '0000' in deadends: return -1
        target = int(target)
        can_go = [True for _ in range(10000)]
        visted = [False for _ in range(10000)]
        for d in deadends:
            can_go[int(d)] = False

        arounds = [1, -1]

        queue = [0]
        visted[0] = True
        ans = 0
        while queue:
            for q in queue:
                if q == target:
                    return ans

            ans += 1
            temp = []
            for q in queue:
                for i in range(8):
                    # 获取下一个可能的位置
                    digits = [q // 1000, q % 1000 // 100, (q % 100) // 10, q % 10]
                    digits[i//2] = (digits[i//2] + arounds[i%2]) % 10  # 某位数字加减1
                    a, b, c, d = digits
                    nxt = a * 1000 + b * 100 + c * 10 + d
                    if can_go[nxt] and not visted[nxt]:
                        visted[nxt] = True
                        temp.append(nxt)

            queue = temp

        return -1
      
```





