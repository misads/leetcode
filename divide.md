# 分治法

## A23. 合并K个排序链表

难度`困难`

#### 题目描述

合并 *k* 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。

> **示例:**

```
输入:
[
  1->4->5,
  1->3->4,
  2->6
]
输出: 1->1->2->3->4->4->5->6
```


#### 题目链接

<https://leetcode-cn.com/problems/merge-k-sorted-lists/>

#### **思路:**

　　分治法，当只有一个链表时直接返回。否则：  

　　① 合并左半边的链表，记为`left`；  

　　② 合并右半边的链表，记为`right`；  

　　③ 合并`left`和`right`(参考[合并两个有序链表](/实用模板?id=合并两个有序链表递归))。  

　　时间复杂度`O(nlog(k))`，`n`是所有链表中元素的总和，`k`是链表个数。  

#### **代码:**

```python
class Solution:
    def merge(self, l1: ListNode, l2: ListNode) -> ListNode:  # 合并两个链表
        if not l1: return l2
        if not l2: return l1
        if l1.val <= l2.val:
            l1.next = self.merge(l1.next, l2)
            return l1
        else:
            l2.next = self.merge(l2.next, l1)
            return l2

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        n = len(lists)
        if not n:
            return

        if n == 1:  # 只有一个，不用合并
            return lists[0]

        mid = n // 2  # 至少为1
        left = self.mergeKLists(lists[:mid])
        right = self.mergeKLists(lists[mid:])
        return self.merge(left, right)

```

## A215. 数组中的第K个最大元素

难度`中等`

#### 题目描述

在未排序的数组中找到第 **k** 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

> **示例 1:**

```
输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
```

> **示例 2:**

```
输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
```

**说明:**

你可以假设 k 总是有效的，且 1 ≤ k ≤ 数组的长度。

#### 题目链接

<https://leetcode-cn.com/problems/kth-largest-element-in-an-array/>

#### **思路:**

　　**方法一：**最大堆，构建以后连续取出`k`个元素即可。  

　　**方法二：**使用快速排序的`partition`函数，`partition`后返回的位置`store_index`，左边的元素一定比它小，右边的元素一定比它大。比较`store_index`和`k`，再对`k`所在的那半边继续`partition`即可。  

#### **代码:**

　　**方法一：**

```python
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heapq.heapify(nums)
        n = len(nums)
        k = n - k + 1  # 转成第k小来做
        
        peek = 0
        for _ in range(k):
            peek = heapq.heappop(nums)
            
        return peek

```

　　**方法二：**

```python
import random

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partition(left, right, pivot_index):
            pivot = nums[pivot_index]
            nums[pivot_index],nums[right] = nums[right], nums[pivot_index]

            store_index = left
            for i in range(left, right):
                if nums[i] < pivot:
                    nums[store_index], nums[i] = nums[i], nums[store_index]
                    store_index +=1
            nums[right], nums[store_index] = nums[store_index], nums[right]

            return store_index

        left = 0
        right = len(nums) - 1
        while left <= right:
            if left == right:
                return nums[left]
            p = random.randint(left, right)
            ans = partition(left, right, p)
            if ans  == len(nums) - k:
                return nums[ans]
            elif ans < len(nums) - k:
                left = ans + 1
            elif ans > len(nums) - k:
                right = ans - 1

```

## A241. 为运算表达式设计优先级

难度`中等`

#### 题目描述

给定一个含有数字和运算符的字符串，为表达式添加括号，改变其运算优先级以求出不同的结果。你需要给出所有可能的组合的结果。有效的运算符号包含 `+`, `-` 以及 `*` 。

> **示例 1:**

```
输入: "2-1-1"
输出: [0, 2]
解释: 
((2-1)-1) = 0 
(2-(1-1)) = 2
```

> **示例 2:**

```
输入: "2*3-4*5"
输出: [-34, -14, -10, -10, 10]
解释: 
(2*(3-(4*5))) = -34 
((2*3)-(4*5)) = -14 
((2*(3-4))*5) = -10 
(2*((3-4)*5)) = -10 
(((2*3)-4)*5) = 10
```


#### 题目链接

<https://leetcode-cn.com/problems/different-ways-to-add-parentheses/>

#### **思路:**

　　从每个运算符的位置都可以拆开，给左右分别加括号。`示例2`的拆分方式如下：  

```python　　　
(2)*(3-4*5)  # 从*拆开
(2*3)-(4*5)  # 从-拆开
(2*3-4)*(5)  # 从*拆开
```

　　对于拆分过的部分，如果还有运算符，可以递归地继续进行拆分。  

#### **代码:**

```python
class Solution:
    def diffWaysToCompute(self, input: str) -> List[int]:
        if not input:
            return []
        if input.isdigit():
            return [int(input)]

        def op(a, b, sym):
            if sym == '+': return a + b
            if sym == '-': return a - b
            if sym == '*': return a * b

        # 每个符号拆开来，两边算
        ans = []
        for i, char in enumerate(input):
            if char in ('+', '-', '*'):
                left = self.diffWaysToCompute(input[:i])
                right = self.diffWaysToCompute(input[i+1:])
                for l in left:
                    for r in right:
                        ans.append(op(l, r, char))

        return ans

```

## A282. 给表达式添加运算符

难度`困难`

#### 题目描述

给定一个仅包含数字 `0-9` 的字符串和一个目标值，在数字之间添加**二元**运算符（不是一元）`+`、`-` 或 `*` ，返回所有能够得到目标值的表达式。

> **示例 1:**

```
输入: num = "123", target = 6
输出: ["1+2+3", "1*2*3"] 
```

> **示例 2:**

```
输入: num = "232", target = 8
输出: ["2*3+2", "2+3*2"]
```

> **示例 3:**

```
输入: num = "105", target = 5
输出: ["1*0+5","10-5"]
```

> **示例 4:**

```
输入: num = "00", target = 0
输出: ["0+0", "0-0", "0*0"]
```

> **示例 5:**

```
输入: num = "3456237490", target = 9191
输出: []
```


#### 题目链接

<https://leetcode-cn.com/problems/expression-add-operators/>

#### **思路:**

　　在`num`中的每一个位置都可以添加 `+-*`号，(也可以不添加)。让我们来找一下规律：  

　　例如`num = "321"`，`target = 7`，在`3`后面添加符号：  

　　① 添加`"+"`，变成了`3 + handle("21") = 7`，也就是解决`handle("21") = 4`这个子问题；  

　　② 添加`"-"`，变成了`3 - handle("21") = 7`，减法其实也可以看成是加法(加上一个负数)，将`handle`前面的负号作为参数传递给`handle`即可，也就是解决`handle("21", factor=-1) = 4`这个子问题；  

　　③ 添加`"*"`，变成了`3 * handle("21") = 7`，需要让`handle`知道有个前导的系数`3`，因此需要解决`handle("21", factor=3) = 7`这个子问题。  

　　④ 不添加任何符号，如果`num * factor`就是`target`也可以符合条件。  

　　**如何解决带`factor`的子问题？**

　　例如`handle("21", factor=3) = 7`，同样可以对应上面的的①②③④。  

　　① 添加`"+"`，变成了`6 + handle("1") = 7`，这里的`6`是`factor * 添加的符号前面的数`得到的；  

　　② 添加`"-"`，变成了`6 - handle("1") = 7`，也就是`handle("1", factor=-1) = 1`；  

　　③ 添加`"*"`，变成了`6 * handle("1") = 7`，也就是`handle("1", factor=6) = 7`；  

　　④ 不添加符号，`3 * 21 != 7`。  

　　显然②③④的子问题都是无解的。  

#### **代码:**

```python
class Solution:
    def myaddOperators(self, num: str, target: int, factor) -> List[str]:
        # 减法其实也可以看成是加法(加上一个负数)
        # factor 表示前一个数的系数， 加法是1，减法是-1，乘法是乘数

        n = len(num)

        ans = []
        if num.isdigit() and int(num) * factor == target:  # 整个可以匹配
            if len(num) == 1 or not num.startswith('0'):  # 不能有前导0
                ans.append(num)
        
        if n == 1:return ans

        for i in range(1, n):
            if i > 1 and num.startswith('0'):  # 不能有前导0
                continue

            current = int(num[:i]) * factor
            
            plus = self.myaddOperators(num[i:], target - current, 1)  #  加法
            for exp in plus:
                ans.append(num[:i] + '+' + exp)

            minus = self.myaddOperators(num[i:], target - current, -1)  #  减法
            for exp in minus:
                ans.append(num[:i] + '-' + exp)

            multiply = self.myaddOperators(num[i:], target, current)  #  乘法
            for exp in multiply:
                ans.append(num[:i] + '*' + exp)

        return ans  


    def addOperators(self, num: str, target: int) -> List[str]:
        return self.myaddOperators(num, target, 1)  # 初始的factor是1

```

## A514. 自由之路

难度`困难`

#### 题目描述

视频游戏“辐射4”中，任务“通向自由”要求玩家到达名为“Freedom Trail Ring”的金属表盘，并使用表盘拼写特定关键词才能开门。

给定一个字符串 **ring**，表示刻在外环上的编码；给定另一个字符串 **key**，表示需要拼写的关键词。您需要算出能够拼写关键词中所有字符的**最少**步数。

最初，**ring** 的第一个字符与12:00方向对齐。您需要顺时针或逆时针旋转 ring 以使 **key** 的一个字符在 12:00 方向对齐，然后按下中心按钮，以此逐个拼写完 **key** 中的所有字符。

旋转 **ring** 拼出 key 字符 **key[i]** 的阶段中：

1. 您可以将 **ring** 顺时针或逆时针旋转**一个位置**，计为1步。旋转的最终目的是将字符串 **ring** 的一个字符与 12:00 方向对齐，并且这个字符必须等于字符 **key[i] 。**
2. 如果字符 **key[i]** 已经对齐到12:00方向，您需要按下中心按钮进行拼写，这也将算作 **1 步**。按完之后，您可以开始拼写 **key** 的下一个字符（下一阶段）, 直至完成所有拼写。

> **示例：**  

　　<img src="_img/514.jpg" style="zoom:40%"/>

```
输入: ring = "godding", key = "gd"
输出: 4
解释:
 对于 key 的第一个字符 'g'，已经在正确的位置, 我们只需要1步来拼写这个字符。 
 对于 key 的第二个字符 'd'，我们需要逆时针旋转 ring "godding" 2步使它变成 "ddinggo"。
 当然, 我们还需要1步进行拼写。
 因此最终的输出是 4。
```

**提示：**

1. **ring** 和 **key** 的字符串长度取值范围均为 1 至 100；
2. 两个字符串中都只有小写字符，并且均可能存在重复字符；
3. 字符串 **key** 一定可以由字符串 **ring** 旋转拼出。


#### 题目链接

<https://leetcode-cn.com/problems/freedom-trail/>

#### **思路:**

　　**方法一：**暴力(超时)，给定`key`中的一个字母`char`时，要么选取向右转的第一个`char`，要么选取向左转的第一个`char`。将移动的次数再加上`key`的长度(需要按按钮的次数)就是结果。最坏时间复杂度为`O(m*2^n)`，其中`n`表示`key`的长度。  
　　**方法二：**动态规划，指数级的复杂度显然是太高了。如果我们已经拼出`key[j]`之前所有字母所需的最小步骤`dp[i]`（这里`i`表示`ring`中的位置），那么拼出`key[j]`的最小步骤`dp[i_]`为 min(`dp[i]` + 从`i`移动到`i_`的最短距离)。  

　　例如，`ring = "cadcdeab"`，`key = "ade"`。

　　对于`key`的第一个字母`a`，可以计算出`dp[1] = 1`，`dp[6] = 2`；  

　　对于`key`的第二个字母`d`，可以计算出`dp'[2] = min(dp[1] + 1, dp[6] + 4) = 2`，`dp'[4] = min(dp[1] + 3, dp[6] + 2) = 4`；    

　　对于`key`的第三个字母`e`，可以计算出`dp''[5] = min(dp'[2] + 3, dp'[4] + 1) = 5`。  

#### **代码:**

　　**方法一：**暴力(超时)

```python
class Solution:
    def myfindRotateSteps(self, ring: str, key: str, cur) -> int:
        if not key:
            return 0
        if ring[cur] == key[0]:
            return self.myfindRotateSteps(ring, key[1:], cur)

        char = key[0]
        right = ring.find(char, cur)
        right_move = right - cur
        if right < 0:
            right = ring.find(char)
            right_move = len(ring) - cur + right

        left = ring.rfind(char, 0, cur)
        left_move = cur - left
        if left < 0:
            left = ring.rfind(char)
            left_move = len(ring) - left + cur

        if left == right:  # 往左转和往右转到达一样的结果
            return min(left_move, right_move) + self.myfindRotateSteps(ring, key[1:], left)
        else:
            l = left_move + self.myfindRotateSteps(ring, key[1:], left)
            r = right_move + self.myfindRotateSteps(ring, key[1:], right)
            return min(l, r)

    def findRotateSteps(self, ring: str, key: str) -> int:
        return self.myfindRotateSteps(ring, key, 0) + len(key)

```

　　**方法二：**动态规划(144ms)

```python
from collections import defaultdict 
class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        mem = collections.defaultdict(list)
        for i, ch in enumerate(ring):
            mem[ch].append(i)
        lr = len(ring)
        lk = len(key)
        dp = [(0, 0)]

        for j, char in enumerate(key):
            temp = []
            for pos in mem[char]:
                minimal = float('inf')
                for last_pos, steps in dp:
                    move = abs(pos - last_pos)
                    minimal = min(minimal, steps + move, steps + lr - move)

                temp.append((pos, minimal))

            dp = temp

        return min([d[1] for d in dp]) + lk
      
```

## A1569. 将子数组重新排序得到同一个二叉查找树的方案数

难度`困难`

#### 题目描述

给你一个数组 `nums` 表示 `1` 到 `n` 的一个排列。我们按照元素在 `nums` 中的顺序依次插入一个初始为空的二叉查找树（BST）。请你统计将 `nums` 重新排序后，统计满足如下条件的方案数：重排后得到的二叉查找树与 `nums` 原本数字顺序得到的二叉查找树相同。

比方说，给你 `nums = [2,1,3]`，我们得到一棵 2 为根，1 为左孩子，3 为右孩子的树。数组 `[2,3,1]` 也能得到相同的 BST，但 `[3,2,1]` 会得到一棵不同的 BST 。

请你返回重排 `nums` 后，与原数组 `nums` 得到相同二叉查找树的方案数。

由于答案可能会很大，请将结果对 `10^9 + 7` 取余数。

> **示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/08/30/bb.png)

```
输入：nums = [2,1,3]
输出：1
解释：我们将 nums 重排， [2,3,1] 能得到相同的 BST 。没有其他得到相同 BST 的方案了。
```

> **示例 2：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/08/30/ex1.png)**

```
输入：nums = [3,4,5,1,2]
输出：5
解释：下面 5 个数组会得到相同的 BST：
[3,1,2,4,5]
[3,1,4,2,5]
[3,1,4,5,2]
[3,4,1,2,5]
[3,4,1,5,2]
```

> **示例 3：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/08/30/ex4.png)**

```
输入：nums = [1,2,3]
输出：0
解释：没有别的排列顺序能得到相同的 BST 。
```

> **示例 4：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/08/30/abc.png)**

```
输入：nums = [3,1,2,5,4,6]
输出：19
```

> **示例  5：**

```
输入：nums = [9,4,2,1,3,6,5,7,8,14,11,10,12,13,16,15,17,18]
输出：216212978
解释：得到相同 BST 的方案数是 3216212999。将它对 10^9 + 7 取余后得到 216212978。
```

**提示：**

- `1 <= nums.length <= 1000`
- `1 <= nums[i] <= nums.length`
- `nums` 中所有数 **互不相同** 。

#### 题目链接

<https://leetcode-cn.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/>

#### **思路:**

　　分治法。第一个元素一定放在根结点；比第一个元素小的元素放在左子树，比第一个元素大的元素放在右子树。  

　　先计算左子树的情况数`left`，再计算右子树的情况数`right`，左右子树可以互相交错排列，只要保持其在各自数组中的位置不变即可，因此总的情况数为`left × right × C(左右子树结点数和, 左子树结点数)`。  

#### **代码:**

```python
class Solution:
    def numOfWays(self, nums: List[int]) -> int:
        def helper(nums):
            if len(nums) <= 2:
                return 1
            root = nums[0]
            less = list(filter(lambda x: x<root, nums))
            more = list(filter(lambda x: x>root, nums))
            return helper(less) * helper(more) * comb(len(nums)-1, len(more))

        return((helper(nums)-1) % 1000000007)

```

