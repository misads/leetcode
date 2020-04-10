# 排序


## A164. 最大间距

难度`困难`

#### 题目描述

给定一个无序的数组，找出数组在排序之后，相邻元素之间最大的差值。

如果数组元素个数小于 2，则返回 0。

> **示例 1:**

```
输入: [3,6,9,1]
输出: 3
解释: 排序后的数组是 [1,3,6,9], 其中相邻元素 (3,6) 和 (6,9) 之间都存在最大差值 3。
```

> **示例 2:**

```
输入: [10]
输出: 0
解释: 数组元素个数小于 2，因此返回 0。
```

**说明:**

- 你可以假设数组中所有元素都是非负整数，且数值在 32 位有符号整数范围内。
- 请尝试在线性时间复杂度和空间复杂度的条件下解决此问题。

#### 题目链接

<https://leetcode-cn.com/problems/maximum-gap/>

#### **思路:**


　　在做这道题目之前，首先学习一下桶排序：

##### **算法过程**

1. 根据待排序集合中最大元素和最小元素的差值范围和映射规则，确定申请的桶个数；
2. 遍历待排序集合，将每一个元素移动到对应的桶中；
3. 对每一个桶中元素进行排序，并移动到已排序集合中。

##### **演示示例**

　　待排序集合为：`[-7, 51, 3, 121, -3, 32, 21, 43, 4, 25, 56, 77, 16, 22, 87, 56, -10, 68, 99, 70]`   

　　映射规则为：

```tex
　　　　\displaystyle f(x)=\frac{x}{10}-c
```

　　其中常量位：(这个公式支持真是太差劲了)  

```tex
　　　　\displaystyle c=\frac{\min }{10}
```

　　即以间隔大小 10 来区分不同值域。

**step 1:**

　　遍历集合可得，最大值为：`max = 121`，最小值为：`min = 10`，待申请桶的个数为：

```tex
　　　　\displaystyle \frac {\max}{10} -\frac {\min}{10}+1=12-(-1)+1=14
```

**step 2:**

　　遍历待排序集合，依次添加各元素到对应的桶中。

| 桶下标 |  桶中元素   |
| ------ | :---------: |
| 0      | -7, -3, -10 |
| 1      |    3, 4     |
| 2      |     16      |
| 3      | 21, 25, 22  |
| 4      |     32      |
| 5      |     43      |
| 6      | 51, 56, 56  |
| 7      |     68      |
| 8      |   77, 70    |
| 9      |     87      |
| 10     |     99      |
| 11     |             |
| 12     |             |
| 13     |     121     |

**step 3:**

对每一个桶中元素进行排序，并移动回原始集合中，即完成排序过程。

##### **算法示例**

```python
def bucketSort(arr):
    maximum, minimum = max(arr), min(arr)
    bucketArr = [[] for i in range(maximum // 10 - minimum // 10 + 1)]  # set the map rule and apply for space
    for i in arr:  # map every element in array to the corresponding bucket
        index = i // 10 - minimum // 10
        bucketArr[index].append(i)
    arr.clear()
    for i in bucketArr:
        heapSort(i)   # sort the elements in every bucket
        arr.extend(i)  # move the sorted elements in bucket to array
```

*引用自<https://www.jianshu.com/p/204ed43aec0c>*



　　回到这一题，有两个关键思路：  

　　① 桶个数的设置，根据抽屉原理，桶的个数为n+1的时候，把n个数放入桶中，一定会产生一个空桶，这样就可以确定最大间距出现在不同的桶之间(跳过空桶)。  

　　② 在每个桶中，并不需要要存放所有的元素，而只要存放该桶中最大的和最小的就行了(因为最大间距一定出现在不同的桶中)。  

#### **代码:**

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0

        minimum, maximum = min(nums), max(nums)
        if maximum == minimum:  # 最大值与最小值相等
            return 0  

        mins = [float('inf')] * (n + 1)  # 存放最小数的桶
        maxs = [float('-inf')] * (n + 1)  # 存放最大数的桶

        for num in nums:
            index = int((num - minimum) * n / (maximum - minimum))
            mins[index] = min(mins[index], num)
            maxs[index] = max(maxs[index], num)

        print(mins)
        print(maxs)
        ans = 0
        m = maxs[0]  # 之前的最大数
        for i in range(1, n + 1):
            if mins[i] < float('inf') and m > float('-inf'):
                ans = max(ans, mins[i] - m)
                m = maxs[i]

        return ans

```

## A179. 最大数

难度`中等`

#### 题目描述

给定一组非负整数，重新排列它们的顺序使之组成一个最大的整数。

> **示例 1:**

```
输入: [10,2]
输出: 210
```

> **示例 2:**

```
输入: [3,30,34,5,9]
输出: 9534330
```

**说明:** 输出结果可能非常大，所以你需要返回一个字符串而不是整数。

#### 题目链接

<https://leetcode-cn.com/problems/largest-number/>

#### **思路:**

　　定义一个比较函数，比较`s1 s2`和`s2 s1`的数值大小，然后用这个比较函数排序。  

#### **代码:**

```python
from functools import cmp_to_key

class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        nums = list(map(str, nums))

        def cmpp(s1, s2):  # 定义一个比较函数
            if int(s1 + s2) > int(s2 + s1):
                return 1
            elif int(s1 + s2) < int(s2 + s1):
                return -1
            else:
                return 0

        nums.sort(key=cmp_to_key(cmpp), reverse=True)
        return str(int(''.join(nums)))  # 去掉前导0

```

## A220. 存在重复元素 III

难度`中等`

#### 题目描述

给定一个整数数组，判断数组中是否有两个不同的索引 *i* 和 *j*，使得 **nums [i]** 和 **nums [j]** 的差的绝对值最大为 *t*，并且 *i* 和 *j* 之间的差的绝对值最大为 *ķ*。

> **示例 1:**

```
输入: nums = [1,2,3,1], k = 3, t = 0
输出: true
```

> **示例 2:**

```
输入: nums = [1,0,1,1], k = 1, t = 2
输出: true
```

> **示例 3:**

```
输入: nums = [1,5,9,1,5,9], k = 2, t = 3
输出: false
```

#### 题目链接

<https://leetcode-cn.com/problems/contains-duplicate-iii/>

#### **思路:**

　　**方法一：**暴力(超时)，用一个`set`记录长度为`k`的窗口中的数字，当窗口长度超过`k`时，去掉第`i - k`个数字(`i`为当前下标)。查找窗口中有没有与当前数字之差的绝对值小于等于`t`的。    
　　**方法二：**如果窗口中的数字是有序的，那么插入和删除的复杂度就可以降低到`O(logn)`，Python中的`bisect`库可以做到这一点。  

#### **代码:**  

　　**方法一：(超时)**

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        if k <= 0 or t < 0:
            return False

        wnd = set()  # 最大为k
        for i, num in enumerate(nums):
            for w in wnd:
                if abs(w-num) <= t:
                    return True
            if len(wnd) >= k:
                wnd.remove(nums[i - k])
            wnd.add(num)
                
        return False

```

　　**方法二：**

```python
import bisect

class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        if k <= 0 or t < 0:
            return False

        wnd = []  # 最大为k
        for i, num in enumerate(nums):
 
            idx = bisect.bisect(wnd, num)
            if 0 <= idx - 1 < len(wnd):
                if num - wnd[idx-1] <= t:
                    return True
            if idx< len(wnd):
               if wnd[idx] - num <= t:
                    return True
            if len(wnd) >= k:
                idx = bisect.bisect_left(wnd, nums[i - k])
                wnd.pop(idx)
                # 上面两行等价于 wnd.remove(nums[i - k]) 不过时间复杂度以O(logn)

            bisect.insort(wnd, num)
                
        return False

```

## A242. 有效的字母异位词

难度`简单`

#### 题目描述

给定两个字符串 *s* 和 *t* ，编写一个函数来判断 *t* 是否是 *s* 的字母异位词。

> **示例 1:**

```
输入: s = "anagram", t = "nagaram"
输出: true
```

> **示例 2:**

```
输入: s = "rat", t = "car"
输出: false
```

**说明:**
你可以假设字符串只包含小写字母。

**进阶:**
如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？

#### 题目链接

<https://leetcode-cn.com/problems/valid-anagram/>

#### **思路:**

　　就是判断两个字符串的字母是否相同。  

#### **代码:**

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        import collections
        cs = collections.Counter(s)  # 字符计数
        ct = collections.Counter(t)
        return cs == ct
      　　
```

　## A324. 摆动排序 II

难度`中等`

#### 题目描述

给定一个无序的数组 `nums`，将它重新排列成 `nums[0] < nums[1] > nums[2] < nums[3]...` 的顺序。

> **示例 1:**

```
输入: nums = [1, 5, 1, 1, 6, 4]
输出: 一个可能的答案是 [1, 4, 1, 5, 1, 6]
```

> **示例 2:**

```
输入: nums = [1, 3, 2, 2, 3, 1]
输出: 一个可能的答案是 [2, 3, 1, 3, 1, 2]
```

**说明:**
你可以假设所有输入都会得到有效的结果。

**进阶:**
你能用 O(n) 时间复杂度和 / 或原地 O(1) 额外空间来实现吗？

#### 题目链接

<https://leetcode-cn.com/problems/wiggle-sort-ii/>

#### **思路:**


　　降序排序以后穿插取数字即可。  

#### **代码:**

```python
class Solution:
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        nums.sort(reverse=True)
        nums[::2], nums[1::2] = nums[len(nums)//2:], nums[:len(nums)//2]

```

## A493. 翻转对

难度`困难`

#### 题目描述

给定一个数组 `nums` ，如果 `i < j` 且 `nums[i] > 2*nums[j]` 我们就将 `(i, j)` 称作一个**重要翻转对**。

你需要返回给定数组中的重要翻转对的数量。

> **示例 1:**

```
输入: [1,3,2,3,1]
输出: 2
```

> **示例 2:**

```
输入: [2,4,3,5,1]
输出: 3
```

**注意:**

1. 给定数组的长度不会超过`50000`。
2. 输入数组中的所有数字都在32位整数的表示范围内。

#### 题目链接

<https://leetcode-cn.com/problems/reverse-pairs/>

#### **思路:**

　　**方法一：**维护一个升序数组，对于一个新的数字`num`，在升序数组中查询`2*num`的位置，然后将`num`插入到升序数组中。  
　　**方法二：**归并排序，递归地计算翻转对的个数，即①先计算左半边 ②再计算右半边 ③用一个两重的`for`循环计算两边的翻转对个数。  

#### **代码:**

　　**方法一：**(bisect)(4000ms)

```python
import bisect

class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        asc = []
        ans = 0
        for i, num in enumerate(nums):
            idx = bisect.bisect_right(asc, num * 2)
            ans += len(asc) - idx
            bisect.insort(asc, num)

        return ans

```

　　**方法二：**(归并排序)(2000ms)

```python

class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        def merge(nums, start, end):
            m = (start + end) // 2
            left, right = nums[start:m + 1], nums[m + 1:end + 1]
            left.append(float('inf'))
            right.append(float('inf'))

            il, ir = 0, 0
            for i in range(start, end + 1):
                if left[il] < right[ir]:
                    nums[i] = left[il]
                    il += 1
                else:
                    nums[i] = right[ir]
                    ir += 1

        def merge_sort_count(nums, start, end) -> int:
            if start >= end: return 0

            m = (start + end) // 2
            c1 = merge_sort_count(nums, start, m)
            c2 = merge_sort_count(nums, m + 1, end)
            ans = c1 + c2
            j = m + 1
            for i in range(start, m + 1):
                while j <= end and nums[i] > 2 * nums[j]:
                    j += 1
                ans += j - (m + 1)

            merge(nums, start, end)
            return ans
        return merge_sort_count(nums, 0, len(nums) - 1)

```

## A524. 通过删除字母匹配到字典里最长单词

难度`中等`

#### 题目描述

给定一个字符串和一个字符串字典，找到字典里面最长的字符串，该字符串可以通过删除给定字符串的某些字符来得到。如果答案不止一个，返回长度最长且字典顺序最小的字符串。如果答案不存在，则返回空字符串。

> **示例 1:**

```
输入:
s = "abpcplea", d = ["ale","apple","monkey","plea"]

输出: 
"apple"
```

> **示例 2:**

```
输入:
s = "abpcplea", d = ["a","b","c"]

输出: 
"a"
```

**说明:**

1. 所有输入的字符串只包含小写字母。
2. 字典的大小不会超过 1000。
3. 所有输入的字符串长度不会超过 1000。

#### 题目链接

<https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting/>

#### **思路:**

　　① 先将d中的单词按照长度由长到短排序，再按照字母顺序，从小到大(用`sort`的`key`功能)。  
　　② 然后依次比较d中字符串与字典s，如果出现匹配就直接返回。

#### **代码:**

```python
class Solution:
    def findLongestWord(self, s: str, d: List[str]) -> str:
        # 先将d中的单词按照长度由长到短排序，再按照字母顺序，从小到大
        d.sort(key=lambda kv: (-len(kv), kv))

        # 然后依次比较d中字符串与字典s，如果出现匹配就直接返回
        for word in d:
            start = 0
            for char in word:
                start = s.find(char, start) + 1
                if not start:
                    break
            else:
                return word

        return ''

```

## A767. 重构字符串

难度`中等`

#### 题目描述

给定一个字符串`S`，检查是否能重新排布其中的字母，使得两相邻的字符不同。

若可行，输出任意可行的结果。若不可行，返回空字符串。

> **示例 1:**

```
输入: S = "aab"
输出: "aba"
```

> **示例 2:**

```
输入: S = "aaab"
输出: ""
```

**注意:**

- `S` 只包含小写字母并且长度在`[1, 500]`区间内。

#### 题目链接

<https://leetcode-cn.com/problems/reorganize-string/>

#### **思路:**

　　判断解存在很容易，`出现次数最多的字符`不能超过总字符数一半[向上取整]，即`max(shown) <= (n+1)//2`。(比如有10个`"a"`和9个其他字母，把其他字母间隔地插入`"a"`中即可)  
　　选择字符使用**大顶堆**，每次都选择出现次数最多的字符，如果这个字符刚被选过，就重新再选一个，并把使用的字符出现次数`-1`，再重新插入到大根堆中。  

#### **代码:**

```python
import collections
import heapq

class Solution:
    def reorganizeString(self, S: str) -> str:
        ls = len(S)
        if not ls:
            return ''
        maximum = (ls + 1) // 2  # 最多的字母个数不能超过这个
        c = collections.Counter(S)
        if max(c.values()) > maximum:  # 最多的字母出现次数大于了一半[上取整]
            return ''

        heap = []
        for key in c:  # 构建初始的堆
            heapq.heappush(heap, (-c[key], key))
        
        ans = ''
        for _ in range(ls):  # 总共要选ls个字符
            counts, letter = heapq.heappop(heap)
            if len(ans) and letter == ans[-1]:  # 用过了, 换一个字母
                counts2, letter2 = heapq.heappop(heap)
                ans += letter2
                heapq.heappush(heap, (counts, letter))
                heapq.heappush(heap, (counts2 + 1, letter2))
            else:
                ans += letter
                heapq.heappush(heap, (counts+1, letter))

        return ans

```

## A853. 车队

难度`中等`

#### 题目描述

`N`  辆车沿着一条车道驶向位于 `target` 英里之外的共同目的地。

每辆车 `i` 以恒定的速度 `speed[i]` （英里/小时），从初始位置 `position[i]` （英里） 沿车道驶向目的地。

一辆车永远不会超过前面的另一辆车，但它可以追上去，并与前车以相同的速度紧接着行驶。

此时，我们会忽略这两辆车之间的距离，也就是说，它们被假定处于相同的位置。

*车队* 是一些由行驶在相同位置、具有相同速度的车组成的非空集合。注意，一辆车也可以是一个车队。

即便一辆车在目的地才赶上了一个车队，它们仍然会被视作是同一个车队。
会有多少车队到达目的地?
> **示例：**

```
输入：target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
输出：3
解释：
从 10 和 8 开始的车会组成一个车队，它们在 12 处相遇。
从 0 处开始的车无法追上其它车，所以它自己就是一个车队。
从 5 和 3 开始的车会组成一个车队，它们在 6 处相遇。
请注意，在到达目的地之前没有其它车会遇到这些车队，所以答案是 3。
```
**提示：**

1. `0 <= N <= 10 ^ 4`
2. `0 < target <= 10 ^ 6`
3. `0 < speed[i] <= 10 ^ 6`
4. `0 <= position[i] < target`
5. 所有车的初始位置各不相同。

#### 题目链接

<https://leetcode-cn.com/problems/car-fleet/>

#### **思路:**

　　因为追上以后会合并，因此**每辆车只可能追上他前面位置的车**。  

　　先判断倒数第二辆车能不能追上最后一辆车，再判断倒数第三辆车能不能追上倒数第二辆车。  

　　以此类推，如果有不能追上的车队数就`+1`。  

　　位置排在最后的车无法追上其他的车，一定是一个车队。  

#### **代码:**

```python
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        n = len(position)
        if n:
            ans = 1  # 最后位置的车
        else:
            return 0
        dict = {position[i]:speed[i] for i in range(n)}

        position = sorted(position, reverse = True)
        
        pre = position[0]
        for nxt in position[1:]:
            time_pre = (target - pre) / dict[pre]
            time_next = (target - nxt) / dict[nxt]
            if time_next <= time_pre:
                continue  # 能追上
            else:
                pre = nxt
                ans += 1

        return ans

```
