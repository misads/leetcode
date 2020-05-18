# 数学

## A169. 多数元素

难度`简单`

#### 题目描述

给定一个大小为 *n* 的数组，找到其中的多数元素。多数元素是指在数组中出现次数**大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

> **示例 1:**

```
输入: [3,2,3]
输出: 3
```

> **示例 2:**

```
输入: [2,2,1,1,1,2,2]
输出: 2
```

#### 题目链接

<https://leetcode-cn.com/problems/majority-element/>

#### **思路:**

　　**方法一：**用一个字典(哈希表)记录每个数出现的次数，如果大于`n//2`则返回该数。  

　　**方法二：**`摩尔投票法`：从第一个数开始`count=1`，遇到相同的就加1，遇到不同的就减1，减到0就重新换个数开始计数，总能找到最多的那个。  

#### **代码:**

　　**方法一**

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        helper = {}
        n = len(nums) // 2
        if n == 0:
            return nums[0]

        for num in nums:
            if num in helper:
                helper[num] += 1
                if helper[num] > n:
                    return num
            else:
                helper[num] = 1
```

　　**方法二**

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 1
        ans = nums[0]
        for i, num in enumerate(nums[1:]):
            if num == ans:
                count += 1
            else:
                count -= 1
                if not count:
                    ans = nums[i+2]

        return ans
```

## A223. 矩形面积

难度`中等`

#### 题目描述

在**二维**平面上计算出两个**由直线构成的**矩形重叠后形成的总面积。

每个矩形由其左下顶点和右上顶点坐标表示，如图所示。

<img src="_img/223.png" style="zoom:100%"/>

> **示例:**

```
输入: -3, 0, 3, 4, 0, -1, 9, 2
输出: 45
```

**说明:** 假设矩形面积不会超出 **int** 的范围。

#### 题目链接

<https://leetcode-cn.com/problems/rectangle-area/>

#### **思路:**

　　两个矩形的并集=`两个矩形的面积和`-`两个矩形的交集`。  

#### **代码:**

```python
class Solution:
    def computeArea(self, A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int) -> int:
        union = (C-A) * (D-B) + (G-E) * (H-F)
        if E >= C or A >= G:
            return union
        if F >= D or B >= H:
            return union
        
        width = min(C, G) - max(A, E)
        # print(width)
        height = min(D, H) - max(B, F)
        # print(height)

        return union - width * height

```

