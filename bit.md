# 位运算

## A78. 子集

难度 `中等`  

#### 题目描述

给定一组**不含重复元素**的整数数组 *nums*，返回该数组所有可能的子集（幂集）。

**说明：**解集不能包含重复的子集。

> **示例:**

```
输入: nums = [1,2,3]
输出:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
```

#### 题目链接

<https://leetcode-cn.com/problems/subsets/>

#### 思路  

　　位运算解法。用`0`到`2^n-1`二进制的`0`和`1`来表示每一位的取或不取。  
　　例如`nums = [1, 2, 3]`。  

| 十进制 | 二进制 | 对应的子集 |
| ------ | ------ | ---------- |
| 0      | 000    | []         |
| 1      | 001    | [3]        |
| 2      | 010    | [2]        |
| 3      | 011    | [2, 3]     |
| 4      | 100    | [1]        |
| 5      | 101    | [1, 3]     |
| 6      | 110    | [1, 2]     |
| 7      | 111    | [1, 2, 3]  |

#### 代码  

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        for i in range(2**n):
            temp = []
            j = 0
            while i != 0:
                if i % 2:
                   temp.append(nums[j]) 
                i = i // 2
                j += 1
            ans.append(temp)

        return ans
```

## A136. 只出现一次的数字

难度`简单`

#### 题目描述

给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

**说明：**

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

> **示例 1:**

```
输入: [2,2,1]
输出: 1
```

> **示例 2:**

```
输入: [4,1,2,1,2]
输出: 4
```

#### 题目链接

<https://leetcode-cn.com/problems/single-number/>

#### **思路:**

　　用位运算**异或**来求解。  

　　已知`0 ^ num = num`，`num ^ num = 0`。  

　　另外，异或运算满足交换律，即`a ^ b ^ c = c ^ a ^ b`。  

　　将所有数取异或，两两相同的元素异或会得到`0`消除掉，结果就是剩下的`只出现一次`的元素。  

#### **代码:**

　　**方法一：**(异或)

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            ans ^= num

        return ans

```

　　**方法二：**(集合)

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        s = set()
        for num in nums:
            if num in s:
                s.remove(num)
            else:
                s.add(num)

        return s.pop()
      
```

## A137. 只出现一次的数字 II

难度`中等`

#### 题目描述

给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。

**说明：**

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

> **示例 1:**

```
输入: [2,2,3,2]
输出: 3
```

> **示例 2:**

```
输入: [0,1,0,1,0,1,99]
输出: 99
```

#### 题目链接

<https://leetcode-cn.com/problems/single-number-ii/>

#### **思路:**

　　计一个状态转换电路，使得一个数出现3次时能自动抵消为0，最后剩下的就是只出现1次的数。  

　　`a`、`b`变量中，相同位置上，分别取出一位，负责完成`00->01->10->00`，当数字出现3次时置零。  

#### **代码:**

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        a = b = 0
        for num in nums:
            b = (b ^ num) & ~a;
            a = (a ^ num) & ~b;
        
        return b;

```

## A201. 数字范围按位与

难度`中等`

#### 题目描述

给定范围 [m, n]，其中 0 <= m <= n <= 2147483647，返回此范围内所有数字的按位与（包含 m, n 两端点）。

> **示例 1:** 

```
输入: [5,7]
输出: 4
```

> **示例 2:**

```
输入: [0,1]
输出: 0
```

#### 题目链接

<https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/>

#### **思路:**

　　`m`和`n`相等时，返回`m`，`m`和`n`不相等时，分别抛弃它们的最低位。  

#### **代码:**

```python
class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        i = 0
        while m != n:
            m >>= 1
            n >>= 1
            i += 1

        return m << i

```

## A318. 最大单词长度乘积

难度`中等`

#### 题目描述

给定一个字符串数组 `words`，找到 `length(word[i]) * length(word[j])` 的最大值，并且这两个单词不含有公共字母。你可以认为每个单词只包含小写字母。如果不存在这样的两个单词，返回 0。

> **示例 1:**

```
输入: ["abcw","baz","foo","bar","xtfn","abcdef"]
输出: 16 
解释: 这两个单词为 "abcw", "xtfn"。
```

> **示例 2:**

```
输入: ["a","ab","abc","d","cd","bcd","abcd"]
输出: 4 
解释: 这两个单词为 "ab", "cd"。
```

> **示例 3:**

```
输入: ["a","aa","aaa","aaaa"]
输出: 0 
解释: 不存在这样的两个单词。
```

#### 题目链接

<https://leetcode-cn.com/problems/maximum-product-of-word-lengths/>

#### **思路:**

　　使用二进制进行状态压缩。26个小写字母可以用一个小于`2^26`的整数来表示。出现过的字母对应位上置1，没出现过的置为0。  

#### **代码:**

```python
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        ans = 0

        hashes = []

        for word in words:
            hash = 0
            for letter in set(word):
                pos = ord(letter) - 97
                hash += 1 << pos

            hashes.append(hash)

        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if hashes[i] & hashes[j] == 0:
                    ans = max(ans, len(word1) * len(word2))

        return ans
```

## A1284. 转化为全零矩阵的最少反转次数

难度`困难`

#### 题目描述

给你一个 `m x n` 的二进制矩阵 `mat`。

每一步，你可以选择一个单元格并将它反转（反转表示 0 变 1 ，1 变 0 ）。如果存在和它相邻的单元格，那么这些相邻的单元格也会被反转。（注：相邻的两个单元格共享同一条边。）

请你返回将矩阵 `mat` 转化为全零矩阵的*最少反转次数*，如果无法转化为全零矩阵，请返回 **-1** 。

二进制矩阵的每一个格子要么是 0 要么是 1 。

全零矩阵是所有格子都为 0 的矩阵。

> **示例 1：**

<img src="_img/1248.png" style="zoom:100%"/>

```
输入：mat = [[0,0],[0,1]]
输出：3
解释：一个可能的解是反转 (1, 0)，然后 (0, 1) ，最后是 (1, 1) 。
```

> **示例 2：**

```
输入：mat = [[0]]
输出：0
解释：给出的矩阵是全零矩阵，所以你不需要改变它。
```

> **示例 3：**

```
输入：mat = [[1,1,1],[1,0,1],[0,0,0]]
输出：6
```

> **示例 4：**

```
输入：mat = [[1,0,0],[1,0,0]]
输出：-1
解释：该矩阵无法转变成全零矩阵
```

**提示：**

- `m == mat.length`
- `n == mat[0].length`
- `1 <= m <= 3`
- `1 <= n <= 3`
- `mat[i][j]` 是 0 或 1 。

#### 题目链接

<https://leetcode-cn.com/problems/minimum-number-of-flips-to-convert-binary-matrix-to-zero-matrix/>

#### **思路:**

　　因此`m`和`n`的取值范围都很小，因此可以借助**状压DP**的思想，将整个矩阵的取值**压缩成一个整数**，如：  

```
[[1 0 1]
 [1 1 1]
 [0 0 1]]
```

　　可以展开成`101111001`，也就是十进制的`337`。  

　　然后将所有可能的反转也用相同的方式压缩成十进制数，如：  

```
[[1 1 0]
 [1 0 0]
 [0 0 0]]
```

　　是一种反转方式，其中`1`表示反转，`0`表示不变，可以展开成`110100000`，十进制为`416`。  

　　我们给某个矩阵应用一种变换，就相当于对这两个整数**做异或操作**。（因为异或0不变，异或1反转）。  

　　使用dp的方式，因为最多只有9个位置，最多翻转9次即可，用`dp[status]`记录翻转到状态`status`需要多少次。当`status=0`时返回结果。  

#### **代码:**

```python
class Solution:
    def minFlips(self, mat: List[List[int]]) -> int:
        m = len(mat)
        n = len(mat[0])
        dp = [float('inf') for _ in range(8**3)]

        def to_line(mat):
            ans = ''
            for line in mat:
                ans += ''.join(map(str, line))

            return int(ans, 2)

        transforms = []
        for i in range(m):
            for j in range(n):
                origin = [[0 for _ in range(n)] for _ in range(m)]
                origin[i][j] = 1
                if i > 0:
                    origin[i-1][j] = 1
                if j > 0:
                    origin[i][j-1] = 1
                if i < m-1:
                    origin[i+1][j] = 1
                if j < n-1:
                    origin[i][j+1] = 1

                transforms.append(to_line(origin))

        idx = to_line(mat)
        dp[idx] = 0
        if idx ==0 :
            return 0

        for step in range(1, 10):  # 最多9步
            dp_ = [float('inf') for _ in range(8 ** 3)]
            for status in range(8**3):
                if dp[status] != float('inf'):
                    for transform in transforms:
                        nxt = status ^ transform
                        if nxt == 0:
                            return step
                        dp_[nxt] = step
            dp = dp_

        return -1

```

