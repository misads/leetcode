# 二分查找

## A50. Pow(x, n)

难度`中等`

#### 题目描述

实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数。

> **示例 1:**

```
输入: 2.00000, 10
输出: 1024.00000
```

> **示例 2:**

```
输入: 2.10000, 3
输出: 9.26100
```

> **示例 3:**

```
输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
```

**说明:**

- -100.0 < *x* < 100.0
- *n* 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。

#### 题目链接

<https://leetcode-cn.com/problems/powx-n/>

#### **思路:**

　　二分法，如果`n`是偶数。则`pow(x, n) = pow(x, n//2) ^ 2`，如果`n`是奇数，则`pow(x, n) = pow(x, n//2) ^ 2 * x`。

　　如果`n`是负数，按`n`的绝对值计算再取倒数即可。  　　

#### **代码:**

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1

        def helper(x, n):
            if n == 1:
                return x
            squre = helper(x, n//2) 
            squre = squre * squre

            if n % 2 == 0:
                return squre
            else:
                return squre * x

        if n < 0:
            return 1 / helper(x, -n)
        else:
            return helper(x, n)

```

## A69. x 的平方根

难度`简单`

#### 题目描述

实现 `int sqrt(int x)` 函数。

计算并返回 *x* 的平方根，其中 *x* 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

> **示例 1:**

```
输入: 4
输出: 2
```

> **示例 2:**

```
输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 
     由于返回类型是整数，小数部分将被舍去。
```

#### 题目链接

<https://leetcode-cn.com/problems/sqrtx/>

#### **思路:**

　　从`1`到`x-1`二分查找。  　　　　

#### **代码:**

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0: return 0
        if x < 4: return 1

        i, j = 2, x - 1
        while i <= i and i < x-1:
            mid = (i+j) // 2
            if mid * mid > x:
                j = mid - 1
            elif mid * mid <= x and (mid + 1) ** 2 > x:
                return mid
            else:
                i = mid + 1
```

## A74. 搜索二维矩阵

难度 `中等`  

#### 题目描述

编写一个高效的算法来判断 *m* x *n* 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

- 每行中的整数从左到右按升序排列。
- 每行的第一个整数大于前一行的最后一个整数。

> **示例 1:**

```
输入:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
输出: true
```

> **示例 2:**

```
输入:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 13
输出: false
```

#### 题目链接

<https://leetcode-cn.com/problems/search-a-2d-matrix/>

#### 思路  

　　整除和取模把一维坐标转为二维，然后套用二分查找模板。时间复杂度`O(log(mn))`。  

#### 代码  

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        if m == 0:
            return False

        n = len(matrix[0])

        def _1d_to_2d(i):
            return i // n, i % n
        
        i, j = 0, m * n
        while i <= j and i < m * n:
            mid = (i + j) // 2
            x, y = _1d_to_2d(mid)
            num = matrix[x][y]
            if num > target:
                j = mid - 1
            elif num < target:
                i = mid + 1
            else:
                return num == target
            
        return False
      
```

## A153. 寻找旋转排序数组中的最小值

难度`中等`

#### 题目描述

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 `[0,1,2,4,5,6,7]` 可能变为 `[4,5,6,7,0,1,2]` )。

请找出其中最小的元素。

你可以假设数组中不存在重复元素。

> **示例 1:**

```
输入: [3,4,5,1,2]
输出: 1
```

> **示例 2:**

```
输入: [4,5,6,7,0,1,2]
输出: 0
```

#### 题目链接

<https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/>

#### **思路:**

　　二分查找，比较第一个数和中间的数，如果中间的数大，则`左半边有序`，否则`右半边有序`。  

　　对于有序的半边，只有第一个数可能是最小的，对于无序的半边继续查找。  　　

#### **代码:**

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        i, j = 0, len(nums) - 1
        ans = float('inf')
        while i <= j and i < len(nums):
            if j - i <= 1:
                return min(ans, min(nums[i:j+1]))
            mid = (i + j) // 2
            if nums[i] < nums[mid]:  # 左边有序
                ans = min(ans, nums[i])
                i = mid + 1
            else:   # 右边有序
                ans = min(ans, nums[mid])
                j = mid - 1

        return ans

```

## A162. 寻找峰值

难度`中等`

#### 题目描述

峰值元素是指其值大于左右相邻值的元素。

给定一个输入数组 `nums`，其中 `nums[i] ≠ nums[i+1]`，找到峰值元素并返回其索引。

数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。

你可以假设 `nums[-1] = nums[n] = -∞`。

> **示例 1:**

```
输入: nums = [1,2,3,1]
输出: 2
解释: 3 是峰值元素，你的函数应该返回其索引 2。
```

> **示例 2:**

```
输入: nums = [1,2,1,3,5,6,4]
输出: 1 或 5 
解释: 你的函数可以返回索引 1，其峰值元素为 2；
     或者返回索引 5， 其峰值元素为 6。
```

**说明:**  

你的解法应该是 *O*(*logN*) 时间复杂度的。  

#### 题目链接

<https://leetcode-cn.com/problems/find-peak-element/>

#### **思路:**

　　要求`O(logN)`复杂度一般考虑二分搜索。该题有如下规律：  

- 规律一：如果`nums[i] > nums[i+1]`，则在`i`之前一定存在峰值元素； 

- 规律二：如果`nums[i] < nums[i+1]`，则在`i+1`之后一定存在峰值元素。

#### **代码:**

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        nums = [float('-inf')] + nums + [float('-inf')]
        i, j = 0, len(nums) - 1
        while i <= j and i < len(nums):
            mid = (i + j) // 2
            if nums[mid] > nums[mid+1]:  # 前半边
                if nums[mid] > nums[mid-1]:
                    return mid - 1
                j = mid - 1
            else:  # 后半边
                i = mid + 1

        return -1
      
```

## A167. 两数之和 II - 输入有序数组

难度`简单`

#### 题目描述

给定一个已按照**升序排列** 的有序数组，找到两个数使得它们相加之和等于目标数。

函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2*。*

**说明:**

- 返回的下标值（index1 和 index2）不是从零开始的。
- 你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

> **示例:**

```
输入: numbers = [2, 7, 11, 15], target = 9
输出: [1,2]
解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
```

#### 题目链接

<https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/>

#### **思路:**

　　注意数组中**可以有负数**。  

　　双指针，初始分别指向第一个元素和最后一个元素，如果它们的和大于`target`，则右指针向左移；如果它们的和小于`target`，则左指针向右移。时间复杂度`O(n)`。  

#### **代码:**

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        nums = numbers
        i, j = 0, len(nums) - 1 

        while i < j:
            to_sum = nums[i] + nums[j]
            if to_sum == target:
                return [i+1, j+1]
            if to_sum > target:
                j -= 1
            elif to_sum <target:
                i += 1
                
```

## A209. 长度最小的子数组

难度`中等`

#### 题目描述

给定一个含有 **n** 个正整数的数组和一个正整数 **s ，**找出该数组中满足其和 **≥ s** 的长度最小的连续子数组**。**如果不存在符合条件的连续子数组，返回 0。

> **示例:** 

```
输入: s = 7, nums = [2,3,1,2,4,3]
输出: 2
解释: 子数组 [4,3] 是该条件下的长度最小的连续子数组。
```

**进阶:**

如果你已经完成了*O*(*n*) 时间复杂度的解法, 请尝试 *O*(*n* log *n*) 时间复杂度的解法。

#### 题目链接

<https://leetcode-cn.com/problems/minimum-size-subarray-sum/>

#### **思路:**

　　双指针。如果`窗口内的数字和`小于`target`，右指针右移，增加数字；如果`窗口内的数字和`大于等于`target`， 左指针**不断右移**(直到窗口内的数字和小于`target`)，然后记录此时窗口的大小。  

#### **代码:**

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        n = len(nums)
        if not n: return 0
        ans = float('inf')
        sum_now = 0
        if max(nums) >= s: return 1  # 最大的数比s大

        left = 0
        for right, num in enumerate(nums):
            sum_now += num
            if sum_now >= s:

                while sum_now >= s:
                    sum_now -= nums[left]
                    left += 1
                # print(left, right)
                ans = min(ans, right - left + 2)


        return ans if ans != float('inf') else 0
      
```

## A222. 完全二叉树的节点个数

难度`中等`

#### 题目描述

给出一个**完全二叉树**，求出该树的节点个数。

**说明：**

[完全二叉树](https://baike.baidu.com/item/完全二叉树/7773232?fr=aladdin)的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。

> **示例:**

```
输入: 
    1
   / \
  2   3
 / \  /
4  5 6

输出: 6
```

#### 题目链接

<https://leetcode-cn.com/problems/count-complete-tree-nodes/>

#### **思路:**

　　层序遍历，按照普通二叉树来统计。  

#### **代码:**

```python
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0

        queue = [root]
        ans = 0
        while queue:
            temp = []
            ans += len(queue)
            # queue存放的是当前层的所有结点
            for q in queue:
                if q.left:
                    temp.append(q.left)
                if q.right:
                    temp.append(q.right)

            queue = temp
        return ans
      
```

## A230. 二叉搜索树中第K小的元素

难度`中等`

#### 题目描述

给定一个二叉搜索树，编写一个函数 `kthSmallest` 来查找其中第 **k** 个最小的元素。

**说明：**
你可以假设 k 总是有效的，1 ≤ k ≤ 二叉搜索树元素个数。

> **示例 1:**

```
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 1
```

> **示例 2:**

```
输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 3
```

**进阶：**
如果二叉搜索树经常被修改（插入/删除操作）并且你需要频繁地查找第 k 小的值，你将如何优化 `kthSmallest` 函数？

#### 题目链接

<https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/>

#### **思路:**

　　二叉搜索树的**中序遍历**是有序的，因此只需要找到中序遍历第`k`个访问的元素即为结果。  

#### **代码:**

　　**写法一：**

```python
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        count = 0  # 访问计数器，每访问一个元素就+1
        def dfs(node):
            nonlocal count
            if not node:
                return None

            ans = dfs(node.left)
            if ans is not None: return ans

            count += 1
            if count == k:
                return node.val
            
            ans = dfs(node.right)
            if ans is not None: return ans

        return dfs(root)
      
```

　　**写法二：**

```python
class Solution:
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        def gen(r):
            if r is not None:
                yield from gen(r.left)
                yield r.val
                yield from gen(r.right)
        
        it = gen(root)
        for _ in range(k):
            ans = next(it)
        return ans
```

## A240. 搜索二维矩阵 II

难度`中等`

#### 题目描述

编写一个高效的算法来搜索 *m* x *n* 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

> **示例:**

现有矩阵 matrix 如下：

```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```

给定 target = `5`，返回 `true`。

给定 target = `20`，返回 `false`。

#### 题目链接

<https://leetcode-cn.com/problems/search-a-2d-matrix-ii/>

#### **思路:**

　　先找到`target`可能在哪几行中，在对这几行分别用**二分法**查找。  

　　如果某一行`line`满足以下两个条件，那么`target`可能在其中：  

　　① 第一个元素小于等于`target`；  

　　② 最后一个元素大于等于`target`。  

　　用二分法找到第一个满足条件②的行，然后从这行开始逐行采用二分法查找`target`，当某一行不满足①时结束。  

#### **代码:**

```python
class Solution:
    def check(self, nums, target):
        i, j = 0, len(nums) - 1
        while i <= j and i < len(nums):
            mid = (i + j) // 2
            if nums[mid] > target:
                j = mid - 1
            elif nums[mid] < target:
                i = mid + 1
            else:
                return True if nums[mid] == target else False
        return False

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        m = len(matrix)
        if not m: return False
        n = len(matrix[0])
        if not n: return False

        i, j = 0, m - 1   # 查行
        while i <= j and i < m:
            mid = (i + j) // 2
            if matrix[mid][0] > target:
                j = mid - 1
            elif matrix[mid][0] < target:
                if mid == m-1 or matrix[mid+1][0] > target:
                    break
                i = mid + 1
            elif matrix[mid][0] == target:
                return True

        for line in range(mid, -1 ,-1):
            if matrix[line][-1] < target:
                break
            if self.check(matrix[line], target):
                return True
            
        return False
      
```

## A275. H指数 II

难度`中等`

#### 题目描述

给定一位研究者论文被引用次数的数组（被引用次数是非负整数），数组已经按照**升序排列**。编写一个方法，计算出研究者的 *h* 指数。

[h 指数的定义](https://baike.baidu.com/item/h-index/3991452?fr=aladdin): “h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （N 篇论文中）**至多**有 h 篇论文分别被引用了**至少** h 次。（其余的 *N - h* 篇论文每篇被引用次数**不多于** *h* 次。）"

> **示例:**

```
输入: citations = [0,1,3,5,6]
输出: 3 
解释: 给定数组表示研究者总共有 5 篇论文，每篇论文相应的被引用了 0, 1, 3, 5, 6 次。
     由于研究者有 3 篇论文每篇至少被引用了 3 次，其余两篇论文每篇被引用不多于 3 次，所以她的 h 指数是 3。
```

**说明:**

如果 *h* 有多有种可能的值 ，*h* 指数是其中最大的那个。

**进阶：**

- 这是 [H指数](https://leetcode-cn.com/problems/h-index/description/) 的延伸题目，本题中的 `citations` 数组是保证有序的。
- 你可以优化你的算法到对数时间复杂度吗？

#### 题目链接

<https://leetcode-cn.com/problems/h-index-ii/>

#### **思路:**

　　一般`O(logn)`的复杂度都使用二分法。为了帮助理解题意，对于示例中的`citations = [0, 1, 3, 5, 6]`，假设有另一个辅助数组`order = [5, 4, 3, 2, 1]`，第一个`citations`大于等于`order`的元素，`order`的值就是结果。  

　　因此二分法的判断条件为比较两个数组的对应元素`citations[i]`和`n - i`。   

#### **代码:**

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        n = len(citations)
        if not n: return 0
        # len(n) ~ 1
        i, j = 0, n - 1
        while i <= j and i < len(citations):
            mid = (i + j) // 2
            if citations[mid] == n - mid:
                return n - mid
            elif citations[mid] < n - mid: 
                i = mid + 1  # 往后搜
            elif citations[mid] > n - mid:
                if mid == 0 or citations[mid-1] < n - (mid-1):
                    return n - mid
                j = mid - 1

        return 0
      
```

## A278. 第一个错误的版本

难度`简单`

#### 题目描述

你是产品经理，目前正在带领一个团队开发新的产品。不幸的是，你的产品的最新版本没有通过质量检测。由于每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本都是错的。

假设你有 `n` 个版本 `[1, 2, ..., n]`，你想找出导致之后所有版本出错的第一个错误的版本。

你可以通过调用 `bool isBadVersion(version)` 接口来判断版本号 `version` 是否在单元测试中出错。实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。

> **示例:**

```
给定 n = 5，并且 version = 4 是第一个错误的版本。

调用 isBadVersion(3) -> false
调用 isBadVersion(5) -> true
调用 isBadVersion(4) -> true

所以，4 是第一个错误的版本。 
```

#### 题目链接

<https://leetcode-cn.com/problems/first-bad-version/>

#### **思路:**

　　二分查找。  

#### **代码:**

```python
class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        i, j = 1, n
        while i <= j and i <= n:
            mid = (i + j) // 2
            if isBadVersion(mid):  # 当前版本是错误的
                if mid == 1 or not isBadVersion(mid-1):  # 前一个版本不是错误的
                    return mid
                j = mid - 1  # 往前找
            else:  # 当前版本是正确的，往后找
                i = mid + 1

        return 1
      
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

## A315. 计算右侧小于当前元素的个数

难度`困难`

#### 题目描述

给定一个整数数组 *nums* ，按要求返回一个新数组 *counts* 。数组 *counts* 有该性质： `counts[i]` 的值是  `nums[i]` 右侧小于 `nums[i]` 的元素的数量。

> **示例:**

```
输入: [5,2,6,1]
输出: [2,1,1,0] 
解释:
5 的右侧有 2 个更小的元素 (2 和 1).
2 的右侧仅有 1 个更小的元素 (1).
6 的右侧有 1 个更小的元素 (1).
1 的右侧有 0 个更小的元素.
```

#### 题目链接

<https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/>

#### **思路:**

　　维护一个升序数组，**从右往左**遍历`nums`，查找`nums[i]`在升序数组中的位置（并插入）就是`counts[i]`。  

　　二分查找时间复杂度`O(logn)`，插入操作`O(n)`，总的时间复杂度为`O(n^2 + nlogn)`。

#### **代码:**

```python
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        n = len(nums)
        if not n: return []
        temp = [nums[-1]]
        ans = [0 for i in range(n)]

        for i in range(n-2, -1, -1):
            num = nums[i]
            idx = bisect.bisect_left(temp, num)
            ans[i] = idx
            bisect.insort(temp, num)

        return ans

```

## A327. 区间和的个数

难度`困难`

#### 题目描述

给定一个整数数组 `nums`，返回区间和在 `[lower, upper]` 之间的个数，包含 `lower` 和 `upper`。
区间和 `S(i, j)` 表示在 `nums` 中，位置从 `i` 到 `j` 的元素之和，包含 `i` 和 `j` (`i` ≤ `j`)。

**说明:**
最直观的算法复杂度是 *O*(*n*2) ，请在此基础上优化你的算法。

> **示例:**

```
输入: nums = [-2,5,-1], lower = -2, upper = 2,
输出: 3 
解释: 3个区间分别是: [0,0], [2,2], [0,2]，它们表示的和分别为: -2, -1, 2。
```

#### 题目链接

<https://leetcode-cn.com/problems/count-of-range-sum/>

#### **思路:**

　　**方法一：**(前缀和+二分)，先计算前n项和`accum(n)`，维护一个升序数组`Asc`将其依次插入。  

　　题目要求的连续区间落在`lower`和`upper`之间，实际上就是求`前n项和`之差落在`lower`和`upper`之间的个数。  

　　插入`accum(n)`时，`Asc`中已经存放着前`1~(n-1)`项和(而且是升序的)，目标是找到其中有几项与`accum(n)`的差落在`lower`和`upper`之间。只要分别(用二分法)查找`accum(n) - upper`和`accum(n) - lower`在`Asc`中的位置，它们的下标差就是满足条件的个数。  

　　**方法二：**权值线段树。python实现起来耗时比方法一更长。  

#### **代码:**

　　**方法一：**(前缀和+二分)

```python
import bisect

class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        n = len(nums)
        if not n: return 0

        ans = 0
        asc = [0]
        temp = 0
        accum = []
        for num in nums:
            temp += num
            accum.append(temp)

        for sum_wnd in accum:
            i = bisect.bisect_left(asc, sum_wnd - upper)  # 查找应该插入的位置
            j = bisect.bisect(asc, sum_wnd - lower)
            bisect.insort(asc, sum_wnd)  # 插入
            ans += j - i

        return ans
```

　　**方法二：**(权值线段树)

```python
import bisect
class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        n = len(nums)
        if not n: return 0
        tree = [0 for _ in range(n+1)]

        ans = 0
        temp = 0
        prefix = []  # 前缀和
        for num in nums:
            temp += num
            prefix.append(temp)

        asc = sorted(prefix)  # 升序的前缀和
        dic = {num :i+1 for i, num in enumerate(asc)}
        rank = [dic[num] for num in prefix]  # 对应的次序

        # ======线段树的三个函数=======
        def lowbit(x):
            return x & (-x)

        def getsum(pos):
            sum = 0
            while pos:
                sum += tree[pos]
                pos -= lowbit(pos)
            return sum

        def update(pos):
            while pos < len(tree):
                tree[pos] += 1
                pos += lowbit(pos)
        # ======线段树的三个函数=======

        for i in range(n):
            if lower <= prefix[i] <= upper: ans += 1
            l = bisect.bisect_left(asc, prefix[i] - upper)  # 查找应该插入的位置
            r = bisect.bisect(asc, prefix[i] - lower)
            ans += getsum(r) - getsum(l);
            update(rank[i])

        return ans

```

## A349. 两个数组的交集

难度`简单`

#### 题目描述

给定两个数组，编写一个函数来计算它们的交集。

> **示例 1:**

```
输入: nums1 = [1,2,2,1], nums2 = [2,2]
输出: [2]
```

> **示例 2:**

```
输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出: [9,4]
```

**说明:**

- 输出结果中的每个元素一定是唯一的。
- 我们可以不考虑输出结果的顺序。

#### 题目链接

<https://leetcode-cn.com/problems/intersection-of-two-arrays/>

#### **思路:**

　　用`set`自带的`&`运算符取交集。  　　　　

#### **代码:**

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return list(set(nums1) & set(nums2))
      
```

## A350. 两个数组的交集 II

难度`简单`

#### 题目描述

给定两个数组，编写一个函数来计算它们的交集。

> **示例 1:**

```
输入: nums1 = [1,2,2,1], nums2 = [2,2]
输出: [2,2]
```

> **示例 2:**

```
输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出: [4,9]
```

**说明：**

- 输出结果中每个元素出现的次数，应与元素在两个数组中出现的次数一致。
- 我们可以不考虑输出结果的顺序。

**进阶:**

- 如果给定的数组已经排好序呢？你将如何优化你的算法？
- 如果 *nums1* 的大小比 *nums2* 小很多，哪种方法更优？
- 如果 *nums2* 的元素存储在磁盘上，磁盘内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？

#### 题目链接

<https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/>

#### **思路:**

　　先取两个数组的交集`inter`，然后每个元素都重复`nums1`和`nums2`中出现次数较少的次数。  

#### **代码:**

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        import collections
        c1 = collections.Counter(nums1)
        c2 = collections.Counter(nums2)
        ans = []
        for k in set(c1.keys()) & set(c2.keys()):
            for _ in range(min(c1[k], c2[k])):
                ans.append(k)

        return ans
      
```

## A352. 将数据流变为多个不相交区间

难度`困难`

#### 题目描述

给定一个非负整数的数据流输入 a1，a2，…，an，…，将到目前为止看到的数字总结为不相交的区间列表。

例如，假设数据流中的整数为 1，3，7，2，6，…，每次的总结为：

```
[1, 1]
[1, 1], [3, 3]
[1, 1], [3, 3], [7, 7]
[1, 3], [7, 7]
[1, 3], [6, 7]
```

**进阶：**
如果有很多合并，并且与数据流的大小相比，不相交区间的数量很小，该怎么办?

**提示：**
特别感谢 [@yunhong](https://discuss.leetcode.com/user/yunhong) 提供了本问题和其测试用例。

#### 题目链接

<https://leetcode-cn.com/problems/data-stream-as-disjoint-intervals/>

#### **思路:**

　　维护一个升序的**不重复的**数组，每次增加新数字时都添加到对应位置。获取`区间总结`时，遍历该数组，将连续出现的数字合并。  

　　也可以用[A57. 插入区间](/array?id=a57-插入区间)的算法。  

#### **代码:**

```python
import bisect

class SummaryRanges:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.asc = []
        self.shown = set()

    def addNum(self, val: int) -> None:
        if val not in self.shown:
            self.shown.add(val)
            bisect.insort_left(self.asc, val)

    def getIntervals(self) -> List[List[int]]:
        asc = self.asc
        if not len(asc): return []
        temp = asc[0]
        ans = []
        start = asc[0]
        for i in range(1, len(self.asc)):
            if self.asc[i] == temp + 1:
                temp += 1
            else:
                ans.append([start, temp])
                start = self.asc[i]
                temp = start

        ans.append([start, temp])
        return ans
      
```

## A367. 有效的完全平方数

难度`简单`

#### 题目描述

给定一个正整数 *num* ，编写一个函数，如果 *num* 是一个完全平方数，则返回 True，否则返回 False。

**说明：**不要使用任何内置的库函数，如  `sqrt`。

> **示例 1：**

```
输入：16
输出：True
```

> **示例 2：**

```
输入：14
输出：False
```

#### 题目链接

<https://leetcode-cn.com/problems/valid-perfect-square/>

#### **思路:**

　　**方法一：**利用[A69. x 的平方根](/binary?id=a69-x-的平方根)的二分法求平方根。  

　　**方法二：**利用 1+3+5+7+9+…+(2n-1)=n^2，即完全平方数肯定是前n个连续奇数的和。  

#### **代码:**

　　**方法一：**

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0: return 0
        if x < 4: return 1

        i, j = 2, x - 1
        while i <= i and i < x-1:
            mid = (i+j) // 2
            if mid * mid > x:
                j = mid - 1
            elif mid * mid <= x and (mid + 1) ** 2 > x:
                return mid
            else:
                i = mid + 1

    def isPerfectSquare(self, num: int) -> bool:
        return (self.mySqrt(num) ** 2 == num)
      
```

　　**方法二：**

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        i = 1
        while num > 0:
            num -= i
            i += 2
        return num == 0
      
```

#### [1283. 使结果不超过阈值的最小除数](https://leetcode-cn.com/problems/find-the-smallest-divisor-given-a-threshold/)

难度中等16收藏分享切换为英文关注反馈

给你一个整数数组 `nums` 和一个正整数 `threshold`  ，你需要选择一个正整数作为除数，然后将数组里每个数都除以它，并对除法结果求和。

请你找出能够使上述结果小于等于阈值 `threshold` 的除数中 **最小** 的那个。

每个数除以除数后都向上取整，比方说 7/3 = 3 ， 10/2 = 5 。

题目保证一定有解。

 

**示例 1：**

```
输入：nums = [1,2,5,9], threshold = 6
输出：5
解释：如果除数为 1 ，我们可以得到和为 17 （1+2+5+9）。
如果除数为 4 ，我们可以得到和为 7 (1+1+2+3) 。如果除数为 5 ，和为 5 (1+1+1+2)。
```

**示例 2：**

```
输入：nums = [2,3,5,7,11], threshold = 11
输出：3
```

**示例 3：**

```
输入：nums = [19], threshold = 5
输出：4
```

 

**提示：**

- `1 <= nums.length <= 5 * 10^4`
- `1 <= nums[i] <= 10^6`
- `nums.length <= threshold <= 10^6`

## A1283. 使结果不超过阈值的最小除数

难度`中等`

#### 题目描述

给你一个整数数组 `nums` 和一个正整数 `threshold`  ，你需要选择一个正整数作为除数，然后将数组里每个数都除以它，并对除法结果求和。

请你找出能够使上述结果小于等于阈值 `threshold` 的除数中 **最小** 的那个。

每个数除以除数后都向上取整，比方说 7/3 = 3 ， 10/2 = 5 。

题目保证一定有解。

> **示例 1：**

```
输入：nums = [1,2,5,9], threshold = 6
输出：5
解释：如果除数为 1 ，我们可以得到和为 17 （1+2+5+9）。
如果除数为 4 ，我们可以得到和为 7 (1+1+2+3) 。如果除数为 5 ，和为 5 (1+1+1+2)。
```

> **示例 2：**

```
输入：nums = [2,3,5,7,11], threshold = 11
输出：3
```

> **示例 3：**

```
输入：nums = [19], threshold = 5
输出：4
```

**提示：**

- `1 <= nums.length <= 5 * 10^4`
- `1 <= nums[i] <= 10^6`
- `nums.length <= threshold <= 10^6`

#### 题目链接

<https://leetcode-cn.com/problems/find-the-smallest-divisor-given-a-threshold/>

#### **思路:**

　　使用二分法在`1`到`max(nums)`的范围内搜索被除数。  

#### **代码:**

```python
class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        def get_divider(divider):
            sum = 0
            for num in nums:
                sum = sum + num // divider if num % divider == 0 else sum + num // divider + 1
                
            return sum
        
        # 范围 1~max(nums)
        maximal = max(nums)
        i, j = 1, maximal
        
        while i <= j and i < maximal:
            mid = (i + j) // 2
            
            if get_divider(mid) > threshold:  
                if mid == maximal or get_divider(mid+1) <= threshold:
                    return mid+1
                
                i = mid + 1  # 要往大了搜
            else:
                j = mid - 1
                
        return mid

```

## ALCP 08. 剧情触发时间

难度`中等`

#### 题目描述

在战略游戏中，玩家往往需要发展自己的势力来触发各种新的剧情。一个势力的主要属性有三种，分别是文明等级（`C`），资源储备（`R`）以及人口数量（`H`）。在游戏开始时（第 0 天），三种属性的值均为 0。

随着游戏进程的进行，每一天玩家的三种属性都会对应**增加**，我们用一个二维数组 `increase` 来表示每天的增加情况。这个二维数组的每个元素是一个长度为 3 的一维数组，例如 `[[1,2,1],[3,4,2]]` 表示第一天三种属性分别增加 `1,2,1` 而第二天分别增加 `3,4,2`。

所有剧情的触发条件也用一个二维数组 `requirements` 表示。这个二维数组的每个元素是一个长度为 3 的一维数组，对于某个剧情的触发条件 `c[i], r[i], h[i]`，如果当前 `C >= c[i]` 且 `R >= r[i]` 且 `H >= h[i]` ，则剧情会被触发。

根据所给信息，请计算每个剧情的触发时间，并以一个数组返回。如果某个剧情不会被触发，则该剧情对应的触发时间为 -1 。

> **示例 1：**

```
输入： `increase = [[2,8,4],[2,5,0],[10,9,8]]requirements = [[2,11,3],[15,10,7],[9,17,12],[8,1,14]]`

输出: `[2,-1,3,-1]`

解释：
初始时，C = 0，R = 0，H = 0
第 1 天，C = 2，R = 8，H = 4
第 2 天，C = 4，R = 13，H = 4，此时触发剧情 0
第 3 天，C = 14，R = 22，H = 12，此时触发剧情 2
剧情 1 和 3 无法触发。
```

> **示例 2：**

```
输入： `increase = [[0,4,5],[4,8,8],[8,6,1],[10,10,0]]` `requirements = [[12,11,16],[20,2,6],[9,2,6],[10,18,3],[8,14,9]]`

输出: `[-1,4,3,3,3]`

示例 3：
输入： `increase = [[1,1,1]]` `requirements = [[0,0,0]]`
输出: `[0]`
```

**限制：**

- `1 <= increase.length <= 10000`
- `1 <= requirements.length <= 100000`
- `0 <= increase[i] <= 10`
- `0 <= requirements[i] <= 100000`

#### 题目链接

<https://leetcode-cn.com/problems/ju-qing-hong-fa-shi-jian/>

#### **思路:**

　　这题因为`requirements`的数据量较大，而`increase`的数据量较小，所以要反过来处理。先用`increase`得到每天的三种属性，然后分别在中间二分查找就可以找到每个需求在哪天可以达成。  

#### **代码:**

```python
import bisect

class Solution:
    def getTriggerTime(self, increase: List[List[int]], requirements: List[List[int]]) -> List[int]:
        mem = {}
        enough = {}
        n = len(increase)
        a, b, c = [0 for _ in range(n+1)], [0 for _ in range(n+1)], [0 for _ in range(n+1)]
        for i, (da, db, dc) in enumerate(increase): 
            a[i+1] = a[i] + da
            b[i+1] = b[i] + db
            c[i+1] = c[i] + dc
            
        ans = []
        for x,y,z in requirements:
            ida = bisect.bisect_left(a, x)
            idb = bisect.bisect_left(b, y)
            idc = bisect.bisect_left(c, z)
            day = max(ida, idb, idc)
            if day == n+1: day = -1
            ans.append(day)
            
        return ans

```

## ALCP 12. 小张刷题计划

难度`中等`

#### 题目描述

为了提高自己的代码能力，小张制定了 `LeetCode` 刷题计划，他选中了 `LeetCode` 题库中的 `n` 道题，编号从 `0` 到 `n-1`，并计划在 `m` 天内**按照题目编号顺序**刷完所有的题目（注意，小张不能用多天完成同一题）。

在小张刷题计划中，小张需要用 `time[i]` 的时间完成编号 `i` 的题目。此外，小张还可以使用场外求助功能，通过询问他的好朋友小杨题目的解法，可以省去该题的做题时间。为了防止“小张刷题计划”变成“小杨刷题计划”，小张每天最多使用一次求助。

我们定义 `m` 天中做题时间最多的一天耗时为 `T`（小杨完成的题目不计入做题总时间）。请你帮小张求出最小的 `T`是多少。

> **示例 1：**

输入：`time = [1,2,3,3], m = 2`

输出：`3`

解释：第一天小张完成前三题，其中第三题找小杨帮忙；第二天完成第四题，并且找小杨帮忙。这样做题时间最多的一天花费了 3 的时间，并且这个值是最小的。

> **示例 2：**

输入：`time = [999,999,999], m = 4`

输出：`0`

解释：在前三天中，小张每天求助小杨一次，这样他可以在三天内完成所有的题目并不花任何时间。

> **限制：**

- `1 <= time.length <= 10^5`
- `1 <= time[i] <= 10000`
- `1 <= m <= 1000`

#### 题目链接

<https://leetcode-cn.com/problems/xiao-zhang-shua-ti-ji-hua/>

#### **思路:**

　　二分法。我们需要找到最小的`T`是多少，最坏情况下`T`的范围在`0~10^9`之间，在这个范围内二分查找即可。  

　　如果给定每天最多的做题时间`T`，小张不可能在`m`天内做完，那么需要增加`T`(在右半边搜)。如果能在`m`天内做完，则减小`T`(在左半边搜)。  

#### **代码:**

```python
class Solution:
    def minTime(self, time: List[int], m: int) -> int:
        if len(time)<=m:
            return 0
        
        def can_T(T):
            day = 1
            accum = 0  # 累积时间
            maximal = 0
            i = 0
            while i < len(time):
                tim = time[i]
                maximal = max(maximal, tim)
                if accum + tim - maximal > T:
                    day += 1
                    maximal = 0  # 最多的求助
                    accum = 0
                    continue
                else:
                    accum += tim

                if day > m:
                    return False

                i += 1
            return True
        
        rr = 1000000000
        i, j = 0, rr
        ans = float('inf')
        while i <= j and i < rr + 1:
            mid = (i + j) // 2
            if can_T(mid):  # 往小了搜
                ans = min(ans, mid)
                j = mid - 1
            else:  # 往大了搜
                i = mid + 1

        return ans

```



