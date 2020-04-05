# 回溯算法

## A37. 解数独

难度`困难`

#### 题目描述

编写一个程序，通过已填充的空格来解决数独问题。

一个数独的解法需**遵循如下规则**：

1. 数字 `1-9` 在每一行只能出现一次。
2. 数字 `1-9` 在每一列只能出现一次。
3. 数字 `1-9` 在每一个以粗实线分隔的 `3x3` 宫内只能出现一次。

空白格用 `'.'` 表示。

<img src="_img/37_1.png" style="zoom:100%"/>

一个数独。

<img src="_img/37_2.png" style="zoom:100%"/>

答案被标成红色。

**Note:**

- 给定的数独序列只包含数字 `1-9` 和字符 `'.'` 。
- 你可以假设给定的数独只有唯一解。
- 给定数独永远是 `9x9` 形式的。

#### 题目链接

<https://leetcode-cn.com/problems/sudoku-solver/>

#### **思路:**

　　标准的dfs。  

#### **代码:**

```python
sys.setrecursionlimit(100000)

class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        fixed = [[False for _ in range(9)] for _ in range(9)]  # 记录原来就有的不能更改的
        row, col, room = [set() for _ in range(9)], [set() for _ in range(9)], [set() for _ in range(9)]  # 用三个集合分别记录每行、每列、每个九宫格用过了哪些数字

        def get_room(i, j):
            return i // 3 * 3 + j // 3

        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    fixed[i][j] = True
                    row[i].add(board[i][j])  # 行
                    col[j].add(board[i][j])  # 列
                    room[get_room(i, j)].add(board[i][j])  # 九宫格

        def dfs(n):  # n取值0 ~ 80，坐标 [n // 9][n % 9]
            while n < 81 and fixed[n // 9][n % 9]:
                n += 1  # 固定的不能修改的

            if n >= 81:
                return True

            x, y = n // 9, n % 9
            for i in range(1, 10):
                element = str(i)
                if element in row[x] or element in col[y] or element in room[get_room(x, y)]:
                    continue  # 这个数字不能用

                row[x].add(element)
                col[y].add(element)
                room[get_room(x, y)].add(element)  
                board[x][y] = str(i)  # (x,y)填上i，然后继续后面的尝试
                if dfs(n + 1):
                    return True
                row[x].remove(element)
                col[y].remove(element)
                room[get_room(x, y)].remove(element)
                board[x][y] = '.'  # 还原现场

            return False

        dfs(0)
```

## A46. 全排列

难度`中等`

#### 题目描述

给定一个 **没有重复** 数字的序列，返回其所有可能的全排列。

> **示例:**

```
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

#### 题目链接

<https://leetcode-cn.com/problems/permutations/>

#### **思路:**

　　dfs。  

#### **代码:**

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        temp = []
        ans = []
        l = len(nums)
        def dfs(n):  # 0~2
            if n > l - 1:
                ans.append(temp.copy())
                return 

            for num in nums:
                if num in temp:
                    continue

                temp.append(num)
                dfs(n+1)
                temp.pop()  # 还原现场
                
        dfs(0)
        return ans
      
```

## A47. 全排列 II

难度`中等`

#### 题目描述

给定一个可包含重复数字的序列，返回所有不重复的全排列。

> **示例:**

```
输入: [1,1,2]
输出:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]
```

#### 题目链接

<https://leetcode-cn.com/problems/permutations-ii/>

#### **思路:**

　　dfs + 集合去重。  

#### **代码:**

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        temp = []
        used = set()  # 使用过的下标
        ans = set()
        l = len(nums)
        def dfs(n):  # 
            if n > l - 1:
                ans.add(tuple(temp))
                return 

            for i in range(l):
                if i in used:
                    continue
                used.add(i)
                temp.append(nums[i])

                dfs(n+1)

                used.remove(i)  # 还原现场
                temp.pop()

        dfs(0)
        return [_ for _ in ans]
      
```

## A51. N皇后

难度`困难`

#### 题目描述

*n* 皇后问题研究的是如何将 *n* 个皇后放置在 *n*×*n* 的棋盘上，并且使皇后彼此之间不能相互攻击。

<img src="_img/51.png" style="zoom:100%"/>

上图为 8 皇后问题的一种解法。

给定一个整数 *n*，返回所有不同的 *n* 皇后问题的解决方案。

每一种解法包含一个明确的 *n* 皇后问题的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

> **示例:**

```
输入: 4
输出: [
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
解释: 4 皇后问题存在两个不同的解法。
```

#### 题目链接

<https://leetcode-cn.com/problems/n-queens/>

#### **思路:**

　　递归。每次考虑一行中的放置位置即可。  

　　放置过程中注意避开其他皇后，即不能在同一列，并且坐标`i+j`和`i-j`都未出现过(斜着互相攻击)。  

#### **代码:**

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        ans = []
        def recur(queens, sum, differ):  # 递归
            row = len(queens)
            if row == n:
                # print(queens)
                ans.append(['.' * q  + 'Q' + '.' * (n-q-1) for q in queens])
                return 

            for i in range(n):  # 处理一行
                if i not in queens and row + i not in sum and row - i not in differ:
                    recur(queens + [i], sum + [row + i], differ + [row - i])
        
        recur([], [], [])
        return ans
      
```

## A52. N皇后 II

难度`困难`

#### 题目描述

*n* 皇后问题研究的是如何将 *n* 个皇后放置在 *n*×*n* 的棋盘上，并且使皇后彼此之间不能相互攻击。

<img src="_img/52.png" style="zoom:100%"/>

上图为 8 皇后问题的一种解法。

给定一个整数 *n*，返回 *n* 皇后不同的解决方案的数量。

> **示例:**

```
输入: 4
输出: 2
解释: 4 皇后问题存在如下两个不同的解法。
[
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
```

#### 题目链接

<https://leetcode-cn.com/problems/n-queens-ii/>

#### **思路:**

　　和上一题一样。  

#### **代码:**

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        ans = 0
        def recur(queens, sum, differ):  # 递归
            nonlocal ans
            row = len(queens)
            if row == n:
                ans += 1
                return 

            for i in range(n):  # 处理一行
                if i not in queens and row + i not in sum and row - i not in differ:
                    recur(queens + [i], sum + [row + i], differ + [row - i])
        
        recur([], [], [])
        return ans
      
```

## A60. 第k个排列

难度`中等`

#### 题目描述

给出集合 `[1,2,3,…,*n*]`，其所有元素共有 *n*! 种排列。

按大小顺序列出所有排列情况，并一一标记，当 *n* = 3 时, 所有排列如下：

1. `"123"`
2. `"132"`
3. `"213"`
4. `"231"`
5. `"312"`
6. `"321"`

给定 *n* 和 *k*，返回第 *k* 个排列。

**说明：**

- 给定 *n* 的范围是 [1, 9]。
- 给定 *k* 的范围是[1,  *n*!]。

> **示例 1:**

```
输入: n = 3, k = 3
输出: "213"
```

> **示例 2:**

```
输入: n = 4, k = 9
输出: "2314"
```

#### 题目链接

<https://leetcode-cn.com/problems/permutation-sequence/>

#### **思路:**

　　相同的第一位有`(n-1)!`种可能，相同的前二位有`(n-2)!`种可能……用整除找出是第几种可能，再到数组中取即可，注意用过的数字要去掉出来。  　　　　

#### **代码:**

```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        factorial = [1 for _ in range(n+1)]  # 阶乘
        for i in range(2, n+1):
            factorial[i] = factorial[i-1] * i

        # n * (n-1) * (n-2).....
        ans = ''

        t = n - 1
        k = k - 1
        set_nums = list(range(1, n+1))

        while t >= 0:
            cur = k // factorial[t]  # 这一位是第几个数字
            ans += str(set_nums[cur])
            set_nums.pop(cur)
            k -= cur * factorial[t]
            t -= 1
        return ans

```

## A77. 组合

难度`中等`

#### 题目描述

给定两个整数 *n* 和 *k*，返回 1 ... *n* 中所有可能的 *k* 个数的组合。

> **示例:**

```
输入: n = 4, k = 2
输出:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

#### 题目链接

<https://leetcode-cn.com/problems/combinations/>

#### **思路:**

　　递归，每次的取值都可以在`上一个数+1`到`n`之间，当取满`k`个时返回。  

#### **代码:**

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        ans = []
        def dfs(i, minimal, nums):  # 第几个数
            if i >= k:
                ans.append(nums.copy())
                return

            for num in range(minimal+1, n+1):  # 保证升序
                dfs(i+1, num, nums + [num])

        dfs(0, 0, [])
        return ans

```

## A89. 格雷编码

难度`中等`

#### 题目描述

格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。

给定一个代表编码总位数的非负整数 *n*，打印其格雷编码序列。格雷编码序列必须以 0 开头。

> **示例 1:**

```
输入: 2
输出: [0,1,3,2]
解释:
00 - 0
01 - 1
11 - 3
10 - 2

对于给定的 n，其格雷编码序列并不唯一。
例如，[0,2,3,1] 也是一个有效的格雷编码序列。

00 - 0
10 - 2
11 - 3
01 - 1
```

> **示例 2:**

```
输入: 0
输出: [0]
解释: 我们定义格雷编码序列必须以 0 开头。
     给定编码总位数为 n 的格雷编码序列，其长度为 2n。当 n = 0 时，长度为 20 = 1。
     因此，当 n = 0 时，其格雷编码序列为 [0]。
```

#### 题目链接

<https://leetcode-cn.com/problems/gray-code/>

#### **思路:**

　　用一个`集合`记录有哪些数字是没有用过的。① 尝试翻转每一位，如果新的数字没有用过就记录下来，然后继续重复 ①。  

　　翻转某一位可以用` ^ (1 << i)`实现，(异或`0`的位不变，异或`1`的位翻转)。　 

#### **代码:**

```python
class Solution:
    def grayCode(self, n: int) -> List[int]:
        not_used = set(range(1, 2**n))
        ans = [0]
        cur = 0
        for _ in range(2**n-1):
            for i in range(n):
                flip = cur ^ (1 << i)  # 翻转某一位
                if flip in not_used:  # 没有使用过
                    cur = flip
                    ans.append(cur)
                    can_use.remove(cur)
                    break  # 跳出里面的for循环，继续下一次①

        return ans
      
```

## A131. 分割回文串

难度`中等`

#### 题目描述

给定一个字符串 *s*，将 *s* 分割成一些子串，使每个子串都是回文串。

返回 *s* 所有可能的分割方案。

> **示例:**

```
输入: "aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]
```

#### 题目链接

<https://leetcode-cn.com/problems/palindrome-partitioning/>

#### **思路:**

　　递归，如果`s`的前`i`位是回文，就对后`n-i`位递归地进行分割，直到分割到空字符串返回。  　　　　

#### **代码:**

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
      
        def recur(s):  # 递归
            if len(s) == 0:  # 空字符串是回文的
                return [[]]

            res = []
            for i in range(1, len(s)+1):
                if s[:i] == s[:i][::-1]:  # s的前i位是回文的
                    for line in recur(s[i:]):
                        res.append([s[:i]] + line)
                        
            return res
          
        return recur(s)
      
```

## A211. 添加与搜索单词 - 数据结构设计

难度`中等`

#### 题目描述

设计一个支持以下两种操作的数据结构：

```
void addWord(word)
bool search(word)
```

search(word) 可以搜索文字或正则表达式字符串，字符串只包含字母 `.` 或 `a-z` 。 `.` 可以表示任何一个字母。

> **示例:**

```
addWord("bad")
addWord("dad")
addWord("mad")
search("pad") -> false
search("bad") -> true
search(".ad") -> true
search("b..") -> true
```

**说明:**

你可以假设所有单词都是由小写字母 `a-z` 组成的。

#### 题目链接

<https://leetcode-cn.com/problems/add-and-search-word-data-structure-design/>

#### **思路:**

　　典型的`trie树`应用，`trie树`(又称字典树或前缀树)，将相同前缀的单词放在同一棵子树上，以实现快速的多对多匹配。如下如所示：  

　　<img src="_img/a211.gif" style="zoom:50%"/>　　　　

　　对于有单词的结点(图中橙色的结点)，我们用一个`"#"`来表示。  

　　上图的`trie树`在Python中的表示是这样的：  

```python
trie = 
{'c': 
    {'o': 
        {'d': {'e': {'#': True}}, 
         'o': {'k': {'#': True}}
         }
     }, 
 'f': 
    {'i': 
        {'v': {'e': {'#': True}}, 
         'l': {'e': {'#': True}}
        }, 
     'a': {'t': {'#': True}}
    }
}
```

　　增加单词时在`Trie树`中插入结点，查找单词时搜索`Trie`树。  

#### **代码:**

```python
class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}  # 声明成员变量

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        # 增加单词
        node = self.trie  
        for char in word:
            node = node.setdefault(char, {})
        node['#'] = True


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        # 搜索trie树
        trie = self.trie
        def recur(n, node):  # n表示模式的第几位 从0开始 
            if n >= len(word):
                return '#' in node  # 匹配串搜索结束，返回trie树对应的结点是否有单词

            char = word[n]
            if char == '.':  # 任意字符
                for nxt in node:  # 下一个
                    if nxt != '#' and recur(n+1, node[nxt]):  # 只能搜字母
                        return True

            else:
                if char in node:
                    return recur(n+1, node[char])

            return False

        return recur(0, trie)

```

## A212. 单词搜索 II

难度`困难`

#### 题目描述

给定一个二维网格 **board** 和一个字典中的单词列表 **words**，找出所有同时在二维网格和字典中出现的单词。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。

> **示例:**

```
输入: 
words = ["oath","pea","eat","rain"] and board =
[
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]

输出: ["eat","oath"]
```

**说明:**
你可以假设所有输入都由小写字母 `a-z` 组成。

**提示:**

- 你需要优化回溯算法以通过更大数据量的测试。你能否早点停止回溯？
- 如果当前单词不存在于所有单词的前缀中，则可以立即停止回溯。什么样的数据结构可以有效地执行这样的操作？散列表是否可行？为什么？ 前缀树如何？如果你想学习如何实现一个基本的前缀树，请先查看这个问题： [实现Trie（前缀树）](https://leetcode-cn.com/problems/implement-trie-prefix-tree/description/)。

#### 题目链接

<https://leetcode-cn.com/problems/word-search-ii/>

#### **思路:**

　　Trie树+dfs搜索。  

　　先用`words`中的单词构建Trie树，然后沿着`board`和trie树同时搜索。当搜索到结束符`"#"`时记录这个单词。  

　　**注意：**搜索到一个单词时要将它从前缀树中删除，否则`board`中再次出现可能会重复。  

　　**优化：**(*摘自官方题解* )

　　在回溯过程中逐渐剪除 Trie 中的节点（剪枝）。   
　　这个想法的动机是整个算法的时间复杂度取决于 Trie 的大小。对于 Trie 中的叶节点，一旦遍历它（即找到匹配的单词），就不需要再遍历它了。结果，我们可以把它从树上剪下来。  

　　逐渐地，这些非叶节点可以成为叶节点以后，因为我们修剪他们的孩子叶节点。在极端情况下，一旦我们找到字典中所有单词的匹配项，Trie 就会变成空的。这个剪枝措施可以减少在线测试用例 50% 的运行时间。  

　　<img src="_img/a212.jpeg" style="zoom:35%"/>　　

#### **代码:**

　　**未优化：**(340ms)

```python
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # words = ['abcd', 'acd', 'ace', 'bc']
        arounds = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ans = []
        trie = {}  # 构造字典树
        for i, word in enumerate(words):
            node = trie
            for char in word:
                node = node.setdefault(char, {})
            node['#'] = i

        m = len(board)
        if not m: return []
        n = len(board[0])

        visted = [[False for _ in range(n)] for _ in range(m)] 
        def dfs(i, j, node):
            if '#' in node:
                ans.append(words[node.pop('#')])  # 查过的单词就去掉

            visted[i][j] = True
            for di, dj in arounds:
                x, y = i + di, j + dj
                if x < 0 or y < 0 or x >= m or y >= n or visted[x][y] or board[x][y] not in node:
                    continue
                dfs(i + di, j + dj, node[board[x][y]])

            visted[i][j] = False  # 还原状态
            #  ①在此处添加剪枝代码
    
        for i in range(m):
            for j in range(n):
                if board[i][j] in trie:
                    dfs(i, j, trie[board[i][j]])


        # print(trie)
        return ans
      
```

　　**剪枝优化：**(280ms)  

```python
# 在①处添加以下剪枝代码，将为空的叶子结点删除
    emptys = []
    for child in node:
        if not node[child]:
            emptys.append(child)

    for em in emptys:
        node.pop(em)
```

## A216. 组合总和 III

难度`中等`

#### 题目描述

找出所有相加之和为 ***n*** 的 **k** 个数的组合**。**组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。

**说明：**

- 所有数字都是正整数。
- 解集不能包含重复的组合。 

> **示例 1:**

```
输入: k = 3, n = 7
输出: [[1,2,4]]
```

> **示例 2:**

```
输入: k = 3, n = 9
输出: [[1,2,6], [1,3,5], [2,3,4]]
```

#### 题目链接

<https://leetcode-cn.com/problems/combination-sum-iii/>

#### **思路:**

　　递归。因为不能包含重复的数字，所以使用**升序**作为答案。每一层递归数字的取值范围在`nums[-1] + 1`到`9`之间。  

#### **代码:**

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        ans = []
        temp = []
        def recur(i, minimal):  # 递归  i <= k
            if i >= k-1:
                left = n - sum(temp)  # 最后一个数用减法，减少一层循环
                if minimal < left <= 9:
                    ans.append(temp.copy()+[left])
                return
            
            for j in range(minimal+1, 10):
                temp.append(j)
                recur(i+1, j)
                temp.pop()

        recur(0, 0)
        return ans
      
```

