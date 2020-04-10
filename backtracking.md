# 回溯算法


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

