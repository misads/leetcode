# BFS(广度优先搜索)

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

<https://leetcode-cn.com/problems/01-matrix/>

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

## A1263. 推箱子

难度`困难`

#### 题目描述

「推箱子」是一款风靡全球的益智小游戏，玩家需要将箱子推到仓库中的目标位置。

游戏地图用大小为 `n * m` 的网格 `grid` 表示，其中每个元素可以是墙、地板或者是箱子。

现在你将作为玩家参与游戏，按规则将箱子 `'B'` 移动到目标位置 `'T'` ：

- 玩家用字符 `'S'` 表示，只要他在地板上，就可以在网格中向上、下、左、右四个方向移动。
- 地板用字符 `'.'` 表示，意味着可以自由行走。
- 墙用字符 `'#'` 表示，意味着障碍物，不能通行。 
- 箱子仅有一个，用字符 `'B'` 表示。相应地，网格上有一个目标位置 `'T'`。
- 玩家需要站在箱子旁边，然后沿着箱子的方向进行移动，此时箱子会被移动到相邻的地板单元格。记作一次「推动」。
- 玩家无法越过箱子。

返回将箱子推到目标位置的最小 **推动** 次数，如果无法做到，请返回 `-1`。

> **示例 1：**

<img src="_img/1263.png" style="zoom:50%"/>

```
输入：grid = [["#","#","#","#","#","#"],
             ["#","T","#","#","#","#"],
             ["#",".",".","B",".","#"],
             ["#",".","#","#",".","#"],
             ["#",".",".",".","S","#"],
             ["#","#","#","#","#","#"]]
输出：3
解释：我们只需要返回推箱子的次数。
```

> **示例 2：**

```
输入：grid = [["#","#","#","#","#","#"],
             ["#","T","#","#","#","#"],
             ["#",".",".","B",".","#"],
             ["#","#","#","#",".","#"],
             ["#",".",".",".","S","#"],
             ["#","#","#","#","#","#"]]
输出：-1
```

> **示例 3：**

```
输入：grid = [["#","#","#","#","#","#"],
             ["#","T",".",".","#","#"],
             ["#",".","#","B",".","#"],
             ["#",".",".",".",".","#"],
             ["#",".",".",".","S","#"],
             ["#","#","#","#","#","#"]]
输出：5
解释：向下、向左、向左、向上再向上。
```

> **示例 4：**

```
输入：grid = [["#","#","#","#","#","#","#"],
             ["#","S","#",".","B","T","#"],
             ["#","#","#","#","#","#","#"]]
输出：-1
```

**提示：**

- `1 <= grid.length <= 20`
- `1 <= grid[i].length <= 20`
- `grid` 仅包含字符 `'.'`, `'#'`,  `'S'` , `'T'`, 以及 `'B'`。
- `grid` 中 `'S'`, `'B'` 和 `'T'` 各只能出现一个。

#### 题目链接

<https://leetcode-cn.com/problems/minimum-moves-to-move-a-box-to-their-target-location/>

#### **思路:**

　　BFS。不过有两种特殊情况需要考虑：  

　　① 人无法到箱子的另一边推箱子，如下图所示：  

```
["#","#","#","#","#","#"]
["#","#","#","#","#","#"]
["#","#",".","B",".","T"]
["#","#","#","#","S","#"]
["#","#","#","#","#","#"]

```

　　② 箱子的位置可能出现重复，如下图所示，先向左推，再向右推：

```
["#","#","#","#","#","#"]
["#",".",".",".","#","#"]
["#",".",".","B",".","T"]
["#",".",".","#","S","#"]
["#","#","#","#","#","#"]

```

　　因此以`(箱子的位置, 人的位置)`保存`visited`数组，箱子每次可以向上下左右移动，不过需要先判断人能不能移动到箱子相反的位置，判断的方式仍然是BFS。因此这道题实际上是双重BFS。  

#### **代码:**

```python
class Solution:
    def minPushBox(self, grid: List[List[str]]) -> int:
        arounds = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        m = len(grid)
        if not m: return 0
        n = len(grid[0])

        box_visited = [[[[False for _ in range(n)] for _ in range(m)] for _ in range(n)] for _ in range(m)]
        
        def can_goto(src, dst):
            visited = [[False for _ in range(n)] for _ in range(m)]
            
            queue = [src]  # (0, 0)
            visited[src[0]][src[1]] = True  
        
            count = 0
            while queue:
                for i, j in queue:
                    if i == dst[0] and j == dst[1]:
                        return True  # 结束的条件

                temp = []
                for i, j in queue:
                    for di, dj in arounds:
                        x, y = i + di, j + dj
                        if x < 0 or y < 0 or x >= m or y >= n or grid[x][y]=='#' or grid[x][y] == 'B':  # 边界
                            continue
                        if not visited[x][y]:
                            visited[x][y] = True
                            temp.append((x, y))

                queue = temp
            return False
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 'S':
                    start = (i, j)
                elif grid[i][j] == 'B':
                    box = (i, j)
                elif grid[i][j] == 'T':
                    target = (i, j)
                    
        queue = [(box[0], box[1], start[0], start[1])]
        grid[box[0]][box[1]] = '.'
        
        count = 0
        while queue:
            for i, j, _, _ in queue:
                if i == target[0] and j == target[1]:
                    return count # 结束的条件

            count += 1

            temp = []
            for i, j, per1, per2 in queue:
                grid[i][j] = '#'
                for di, dj in arounds:
                    x, y = i + di, j + dj  # 箱子的下一个位置
                    if x < 0 or y < 0 or x >= m or y >= n or grid[x][y]=='#':  # 边界
                        continue
                    x_, y_ = i - di, j - dj  # 人需要站在的位置
                    if x_ < 0 or y_ < 0 or x_ >= m or y_ >= n or grid[x][y]=='#':  # 边界
                        continue
                    if not can_goto((per1, per2), (x_, y_)):  # 走不到那个位置
                        continue

                    if not box_visited[x][y][i][j]:  # (箱子的位置，人推完箱子后的位置)
                        box_visited[x][y][i][j] = True 
                        temp.append((x, y, i, j))

                grid[i][j] = '.'
            queue = temp
                
        return -1
```

## ALCP 09. 最小跳跃次数

难度`困难`

#### 题目描述

为了给刷题的同学一些奖励，力扣团队引入了一个弹簧游戏机。游戏机由 `N` 个特殊弹簧排成一排，编号为 `0` 到 `N-1`。初始有一个小球在编号 `0` 的弹簧处。若小球在编号为 `i` 的弹簧处，通过按动弹簧，可以选择把小球向右弹射 `jump[i]` 的距离，或者向左弹射到任意左侧弹簧的位置。也就是说，在编号为 `i` 弹簧处按动弹簧，小球可以弹向 `0` 到 `i-1` 中任意弹簧或者 `i+jump[i]` 的弹簧（若 `i+jump[i]>=N` ，则表示小球弹出了机器）。小球位于编号 0 处的弹簧时不能再向左弹。

为了获得奖励，你需要将小球弹出机器。请求出最少需要按动多少次弹簧，可以将小球从编号 `0` 弹簧弹出整个机器，即向右越过编号 `N-1` 的弹簧。

> **示例 1：**

> 输入：`jump = [2, 5, 1, 1, 1, 1]`
>
> 输出：`3`
>
> 解释：小 Z 最少需要按动 3 次弹簧，小球依次到达的顺序为 0 -> 2 -> 1 -> 6，最终小球弹出了机器。

**限制：**

- `1 <= jump.length <= 10^6`
- `1 <= jump[i] <= 10000`

#### 题目链接

<https://leetcode-cn.com/problems/zui-xiao-tiao-yue-ci-shu/>

#### **思路:**

　　BFS，用`visited`数组记录走过的位置，可以将时间复杂度控制在`O(n)`。  
　　有一个注意点是向左跳不会超过来的时候的位置，比如从2的位置跳到5，向左跳的时候最多跳到3，在BFS的过程中用一个`last`变量记录跳过来时的位置。  

#### **代码:**

```python
class Solution:
    def minJump(self, jump: List[int]) -> int:
        n = len(jump)
        if jump[0] >= n:
            return 1
        visited = [False for _ in range(n)]

        queue = [0]
        visited[0] = 0

        count = 0
        last = 0
        while queue:
            count += 1
            temp = []
            for i in queue:
                x = i + jump[i]
                if x >= n:
                    return count

                if not visited[x]:  # 往后走
                    visited[x] = True
                    temp.append(x)

                for x in range(i - 1, last, -1):  # 往前走
                    if not visited[x]:
                        visited[x] = True
                        temp.append(x)

                last = max(last, i)

            queue = temp

```



