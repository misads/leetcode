# 滑动窗口

## A239. 滑动窗口最大值

难度`困难`

#### 题目描述

给定一个数组 *nums* ，有一个大小为 *k* 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 *k* 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。
**进阶：**

你能在线性时间复杂度内解决此题吗？
> **示例:**

```
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```
**提示：**

- `1 <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`
- `1 <= k <= nums.length`

#### 题目链接

<https://leetcode-cn.com/problems/sliding-window-maximum/>

#### **思路:**

　　**方法一：**维护一个升序数组`Asc`，存放窗口内的数，每次取`Asc`的最后一个元素就是当前窗口的最大值。  

　　用二分法插入新元素和删除滑出窗口的元素，插入和删除的时间复杂度为`O(logn)`。  

　　**方法二：**滑动窗口+最大堆。  

#### **代码:**

　　**方法一：**(升序数组)

```python
import bisect

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)

        asc = []
        for i in range(k):
            bisect.insort(asc, nums[i]) 

        ans = [0 for _ in range(n - k + 1)]
        ans[0] = asc[-1]

        for i in range(k, n):
            idx = bisect.bisect_left(asc, nums[i - k])
            asc.pop(idx)
            bisect.insort(asc, nums[i]) 
            ans[i-k+1] = asc[-1]

        return ans

```

　　**方法二：**(滑动窗口+最大堆)

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        left = 0
        heap = []
        ans = []
        for right, num in enumerate(nums):
            heappush(heap, (-num, right))
            while heap:
                maximal, i = heappop(heap)
                if i > right - k:  # (right-k, right]
                    heappush(heap, (maximal, i))
                    break

            if right >= k - 1:
                ans.append(-maximal)

        return ans
```

## A480. 滑动窗口中位数

难度`困难`

#### 题目描述

中位数是有序序列最中间的那个数。如果序列的大小是偶数，则没有最中间的数；此时中位数是最中间的两个数的平均数。

例如：

- `[2,3,4]`，中位数是 `3`
- `[2,3]`，中位数是 `(2 + 3) / 2 = 2.5`

给你一个数组 *nums* ，有一个大小为 *k* 的窗口从最左端滑动到最右端。窗口中有 *k* 个数，每次窗口向右移动 *1* 位。你的任务是找出每次窗口移动后得到的新窗口中元素的中位数，并输出由它们组成的数组。
> **示例：**

给出 *nums* = `[1,3,-1,-3,5,3,6,7]`，以及 *k* = 3。

```
窗口位置                      中位数
---------------               -----
[1  3  -1] -3  5  3  6  7       1
 1 [3  -1  -3] 5  3  6  7      -1
 1  3 [-1  -3  5] 3  6  7      -1
 1  3  -1 [-3  5  3] 6  7       3
 1  3  -1  -3 [5  3  6] 7       5
 1  3  -1  -3  5 [3  6  7]      6
```

 因此，返回该滑动窗口的中位数数组 `[1,-1,-1,3,5,6]`。
**提示：**

- 你可以假设 `k` 始终有效，即：`k` 始终小于输入的非空数组的元素个数。
- 与真实值误差在 `10 ^ -5` 以内的答案将被视作正确答案。

#### 题目链接

<https://leetcode-cn.com/problems/sliding-window-median/>

#### **思路:**

　　和[A239. 滑动窗口最大值](/sliding?id=a239-滑动窗口最大值)一样，维护一个长度固定为`k`的升序数组`Asc`，每次取它的中位数即为该窗口的中位数。  
　　新数据的插入和出窗口数据的删除都使用二分法完成。  

#### **代码:**

```python
import bisect

class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        n = len(nums)

        asc = []
        for i in range(k):
            bisect.insort(asc, nums[i]) 

        ans = [0 for _ in range(n - k + 1)]

        get_middile = lambda: asc[k // 2] if k % 2 == 1 else 0.5 * (asc[k // 2 - 1] + asc[k // 2])
        ans[0] = get_middile()

        for i in range(k, n):
            idx = bisect.bisect_left(asc, nums[i - k]); asc.pop(idx)
            # 上面一行以O(logn)删除nums[i - k]，相当于asc.remove(nums[i - k])
            bisect.insort(asc, nums[i]) 
            ans[i-k+1] = get_middile()

        return ans

```

## A978. 最长湍流子数组

难度`中等`

#### 题目描述

当 `A` 的子数组 `A[i], A[i+1], ..., A[j]` 满足下列条件时，我们称其为*湍流子数组*：

- 若 `i <= k < j`，当 `k` 为奇数时， `A[k] > A[k+1]`，且当 `k` 为偶数时，`A[k] < A[k+1]`；
- **或** 若 `i <= k < j`，当 `k` 为偶数时，`A[k] > A[k+1]` ，且当 `k` 为奇数时， `A[k] < A[k+1]`。

也就是说，如果比较符号在子数组中的每个相邻元素对之间翻转，则该子数组是湍流子数组。

返回 `A` 的最大湍流子数组的**长度**。
> **示例 1：**

```
输入：[9,4,2,10,7,8,8,1,9]
输出：5
解释：(A[1] > A[2] < A[3] > A[4] < A[5])
```

> **示例 2：**

```
输入：[4,8,12,16]
输出：2
```

> **示例 3：**

```
输入：[100]
输出：1
```
**提示：**

1. `1 <= A.length <= 40000`
2. `0 <= A[i] <= 10^9`

#### 题目链接

<https://leetcode-cn.com/problems/longest-turbulent-subarray/>

#### **思路:**

　　类似于[A376. 摆动序列](/dp?id=a376-摆动序列)，不过这题要求子数组连续。  

　　如果一个数大于前面的数，称为当前是**上升**的，否则称为当前是**下降**的。`asc[i]`表示`A[i] > A[i-1]`时，以`A[i]`结尾的最长湍流子数组，`dsc[i]`表示`A[i] < A[i-1]`时，以`A[i]`结尾的最长湍流子数组。  

　　如果当前是**上升**的，有状态转移方程`asc[i] = dsc[i-1] + 1`；同样，如果当前是**下降**的，有状态转移方程`dsc[i] = asc[i-1] + 1`。  

#### **代码:**

```python
class Solution:
    def maxTurbulenceSize(self, A: List[int]) -> int:
        nums = A
        n = len(nums)
        if n <= 1: return n

        asc = [1 for i in range(n)]  # 上升
        dsc = [1 for i in range(n)]  # 下降  

        ans = 1
        for i in range(1, n):
            if nums[i] > nums[i-1]:  # 某个数大于前面的数
                asc[i] = dsc[i-1] + 1  
                ans = max(ans, asc[i])
            elif nums[i] < nums[i-1]:
                dsc[i] = asc[i-1] + 1
                ans = max(ans, dsc[i])
            else:
                asc[i] = dsc[i] = 1

        return ans

```

## A995. K 连续位的最小翻转次数

难度`困难`

#### 题目描述

在仅包含 `0` 和 `1` 的数组 `A` 中，一次 *K 位翻转*包括选择一个长度为 `K` 的（连续）子数组，同时将子数组中的每个 `0` 更改为 `1`，而每个 `1` 更改为 `0`。

返回所需的 `K` 位翻转的次数，以便数组没有值为 `0` 的元素。如果不可能，返回 `-1`。
> **示例 1：**

```
输入：A = [0,1,0], K = 1
输出：2
解释：先翻转 A[0]，然后翻转 A[2]。
```

> **示例 2：**

```
输入：A = [1,1,0], K = 2
输出：-1
解释：无论我们怎样翻转大小为 2 的子数组，我们都不能使数组变为 [1,1,1]。
```

> **示例 3：**

```
输入：A = [0,0,0,1,0,1,1,0], K = 3
输出：3
解释：
翻转 A[0],A[1],A[2]: A变成 [1,1,1,1,0,1,1,0]
翻转 A[4],A[5],A[6]: A变成 [1,1,1,1,1,0,0,0]
翻转 A[5],A[6],A[7]: A变成 [1,1,1,1,1,1,1,1]
```
**提示：**

1. `1 <= A.length <= 30000`
2. `1 <= K <= A.length`


#### 题目链接

<https://leetcode-cn.com/problems/minimum-number-of-k-consecutive-bit-flips/>

#### **思路:**

　　**方法一：**贪心算法+翻转(超时)，如果数组中`nums[i]`的位置出现了`0`，就对`nums[i: i+k]`都进行翻转，如果`i+k`大于数组的长度，则无论如何操作，都不可能变成全`1`。时间复杂度`O(nk)`，其中n表示数组长度。  

　　**方法二：**记录翻转边界(1160ms)，方法一可能会造成大量重复的翻转，因此不对数组中的元素进行翻转，而是用一个标志`flip`记录当前有没有翻转，每次翻转时都另`flip = 1 - flip`。另外，当前元素超出之前翻转的边界时也要再翻转一次，所以要用一个`boundary`数组记录之前所有的翻转边界。  

#### **代码:**

　　**方法一：**(贪心+翻转数组位, 超时)

```python
class Solution:
    def minKBitFlips(self, A: List[int], K: int) -> int:
        n = len(A)
        ans = 0
        for i, num in enumerate(A):
            if num == 1:
                continue
            if i + K > n:
                return - 1
            ans += 1
            for j in range(i, i + K):
                A[j] = 1 - A[j]
            
        return ans

```

　　**方法二：**(记录翻转边界, 1160ms)

```python
class Solution:
    def minKBitFlips(self, A: List[int], K: int) -> int:
        n = len(A)
        ans = 0
        boundary = []  # 记录所有翻转边界的数组

        flip = 1
        for i, num in enumerate(A):
            if boundary and boundary[0] == i:  # 到了某一次的翻转边界 
                flip = 1 - flip
                boundary.pop(0)

            if num == flip:  # 这个数字不需要翻转了
                continue

            if i + K > n:
                return - 1
            ans += 1

            flip = 1 - flip
            boundary.append(i+K)  # 记录翻转边界，即 i + K
            
        return ans
      
```

## A992. K 个不同整数的子数组

难度`困难`

#### 题目描述

给定一个正整数数组 `A`，如果 `A` 的某个子数组中不同整数的个数恰好为 `K`，则称 `A` 的这个连续、不一定独立的子数组为*好子数组*。

（例如，`[1,2,3,1,2]` 中有 `3` 个不同的整数：`1`，`2`，以及 `3`。）

返回 `A` 中*好子数组*的数目。
> **示例 1：**

```
输入：A = [1,2,1,2,3], K = 2
输出：7
解释：恰好由 2 个不同整数组成的子数组：[1,2], [2,1], [1,2], [2,3], [1,2,1], [2,1,2], [1,2,1,2].
```

> **示例 2：**

```
输入：A = [1,2,1,3,4], K = 3
输出：3
解释：恰好由 3 个不同整数组成的子数组：[1,2,1,3], [2,1,3], [1,3,4].
```
**提示：**

1. `1 <= A.length <= 20000`
2. `1 <= A[i] <= A.length`
3. `1 <= K <= A.length`


#### 题目链接

<https://leetcode-cn.com/problems/subarrays-with-k-different-integers/>

#### **思路:**

　　滑动窗口+中间试探，关键在于**状态还原**。  

　　① 先固定`left`，`right`指针向右试探；  

　　② 满足窗口时，记录结果，同时用`temp`指针从`left`向右试探，看左边减少一些字母后是否仍然能满足条件；  

　　③ 大于窗口时，`left`指针右移，然后重复①。  

#### **代码:**

```python
from collections import defaultdict

class Solution:
    def subarraysWithKDistinct(self, A, K: int) -> int:
        mem = defaultdict(int)  # 记录窗口内每个字符的出现次数
        ans = 0
        left = 0

        for right, num in enumerate(A):
            mem[A[right]] += 1

            while len(mem) > K:
                mem[A[left]] -= 1
                if mem[A[left]] == 0:
                    mem.pop(A[left])
                left += 1

            if len(mem) == K:
                temp = left
                while len(mem) == K:
                    ans += 1
                    mem[A[temp]] -= 1
                    if mem[A[temp]] == 0:
                        mem.pop(A[temp])

                    temp += 1

                while temp > left:
                    mem[A[temp - 1]] += 1
                    temp -= 1

                assert len(mem) == K  # 还原之前的状态

        return ans


```

## A1423. 可获得的最大点数

难度`中等`

#### 题目描述

几张卡牌 **排成一行**，每张卡牌都有一个对应的点数。点数由整数数组 `cardPoints` 给出。

每次行动，你可以从行的开头或者末尾拿一张卡牌，最终你必须正好拿 `k` 张卡牌。

你的点数就是你拿到手中的所有卡牌的点数之和。

给你一个整数数组 `cardPoints` 和整数 `k`，请你返回可以获得的最大点数。

> **示例 1：**

```
输入：cardPoints = [1,2,3,4,5,6,1], k = 3
输出：12
解释：第一次行动，不管拿哪张牌，你的点数总是 1 。但是，先拿最右边的卡牌将会最大化你的可获得点数。最优策略是拿右边的三张牌，最终点数为 1 + 6 + 5 = 12 。
```

> **示例 2：**

```
输入：cardPoints = [2,2,2], k = 2
输出：4
解释：无论你拿起哪两张卡牌，可获得的点数总是 4 。
```

> **示例 3：**

```
输入：cardPoints = [9,7,7,9,7,7,9], k = 7
输出：55
解释：你必须拿起所有卡牌，可以获得的点数为所有卡牌的点数之和。
```

> **示例 4：**

```
输入：cardPoints = [1,1000,1], k = 1
输出：1
解释：你无法拿到中间那张卡牌，所以可以获得的最大点数为 1 。 
```

> **示例 5：**

```
输入：cardPoints = [1,79,80,1,1,1,200,1], k = 3
输出：202
```

**提示：**

- `1 <= cardPoints.length <= 10^5`
- `1 <= cardPoints[i] <= 10^4`
- `1 <= k <= cardPoints.length`

#### 题目链接

<https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards/>

#### **思路:**

　　因为拿掉的牌**固定为**`k`张，因此剩下的就是中间的`n-k`张，维护一个滑动窗口，求中间`n-k`张的最小值即可。  

#### **代码:**

```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n = len(cardPoints) - k
        if n == 0:
            return sum(cardPoints)
        left = 0
        wnd = 0
        ans = inf
        for right, num in enumerate(cardPoints):  # cardPoints[left: right+1]
            wnd += num
                
            if right + 1 - left == n:  # 窗口内的数量正好为n
                ans = min(ans, wnd)
                wnd -= cardPoints[left]  # 减掉一个
                left += 1
            
        return sum(cardPoints) - ans

```

