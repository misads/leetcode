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

## A287. 寻找重复数

难度`中等`

#### 题目描述

给定一个包含 *n* + 1 个整数的数组 *nums* ，其数字都在 1 到 *n* 之间（包括 1 和 *n* ），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。

> **示例 1:**

```
输入: [1,3,4,2,2]
输出: 2
```

> **示例 2:**

```
输入: [3,1,3,4,2]
输出: 3
```

**说明：**

1. **不能**更改原数组（假设数组是只读的）。
2. 只能使用额外的 *O*(1) 的空间。
3. 时间复杂度小于 *O*(*n*2) 。
4. 数组中只有一个重复的数字，但它可能不止重复出现一次。

#### 题目链接

<https://leetcode-cn.com/problems/find-the-duplicate-number/>

#### **思路:**

- 其一，对于链表问题，使用快慢指针可以判断是否有环。
- 其二，本题可以使用数组配合下标，抽象成链表问题。

　　快慢指针第一次重合后，将快指针也变成慢指针，第二次必然在重复数字处重合。  

#### **代码:**

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        fast, slow = 0, 0
        while True:
            fast = nums[nums[fast]];  # 快指针一次走2步
            slow = nums[slow];  # 慢指针一次走1步
            if slow == fast:
                fast = 0
                while slow != fast:
                    fast = nums[fast]
                    slow = nums[slow]

                return slow

```
