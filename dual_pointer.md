# 双指针

## A11. 盛最多水的容器

难度 `中等`

#### 题目描述

给你 *n* 个非负整数 *a*1，*a*2，...，*a*n，每个数代表坐标中的一个点 (*i*, *ai*) 。在坐标内画 *n* 条垂直线，垂直线 *i* 的两个端点分别为 (*i*, *ai*) 和 (*i*, 0)。找出其中的两条线，使得它们与 *x* 轴共同构成的容器可以容纳最多的水。

**说明：**你不能倾斜容器，且 *n* 的值至少为 2。

 

![img](_img/11.jpg)

图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。

 

**示例：**

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49
```

#### 题目链接

<https://leetcode-cn.com/problems/container-with-most-water/>

#### 思路  

　　方法一：用`lefts`记录`heights`从左到右所有比之前最高的还高的线，`rights`记录从右到左所有比之前最高的还高的线。遍历`lefts`和`rights`，其中必有一种组合能够容纳最多的水。  
　　方法二：双指针，初始时设头指针和尾指针分别为`i`和`j`。我们能够发现不管是左指针向右移动一位，还是右指针向左移动一位，容器的底都是一样的，都比原来减少了 1。这种情况下我们想要让指针移动后的容器面积增大，就要使移动后的容器的高尽量大，所以我们选择指针所指的高较小的那个指针进行移动，这样我们就保留了容器较高的那条边，放弃了较小的那条边，以获得有更高的边的机会。

#### 代码  

　　方法一：

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        lefts = [0]
        rights = [len(height)-1]
        tmp = height[0]
        for i, num in enumerate(height):
            if num > tmp: 
                lefts.append(i)
                tmp = num


        tmp = height[-1]
        for i in range(len(height)-1,-1,-1):
            num = height[i]
            if num > tmp: 
                rights.append(i)
                tmp = num

        def calc(i1, i2):
            return (i2-i1) * (min(height[i1],height[i2]))

        l, r = len(lefts), len(rights)
        i, j = 0, 0
        ans = 0

        for ll in lefts:
            for rr in rights:
                temp = calc(ll,rr)
                if temp > ans:
                    ans = temp
        return ans
```

　　方法二：

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l = len(height)
        i, j = 0, l - 1
        ans = 0
        while i < j:
            h = min(height[i], height[j])
            ans = max(ans, h * (j-i))
            # 指针向所指的高较小的那个指针进行移动
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1

        return ans
```

## A75. 颜色分类

难度 `中等`  

#### 题目描述

给定一个包含红色、白色和蓝色，一共 *n* 个元素的数组，**原地**对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

**注意:**
不能使用代码库中的排序函数来解决这道题。

> **示例:**

```
输入: [2,0,2,1,1,0]
输出: [0,0,1,1,2,2]
```

#### 题目链接

<https://leetcode-cn.com/problems/sort-colors/>

#### 思路  

　　使用三个指针可以实现**一趟扫描**，`low`的前面全都是`0`，而`high`的后面全都是`2`。`i`表示当前位置，如果当前位置为`0`，则与`low`交换。如果为`2`，则与`high`交换。  

#### 代码  

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        low, high = 0, len(nums) - 1
        i = 0
        while i <= high:
            if nums[i] == 0:
                nums[i], nums[low] = nums[low], nums[i]
                low += 1
                i += 1
            elif nums[i] == 2:
                nums[i], nums[high] = nums[high], nums[i]
                high -= 1
            elif nums[i] == 1:
                i += 1

```

## A283. 移动零

难度`简单`

#### 题目描述

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

> **示例:**

```
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
```

**说明**:

1. 必须在原数组上操作，不能拷贝额外的数组。
2. 尽量减少操作次数。

#### 题目链接

<https://leetcode-cn.com/problems/move-zeroes/>

#### **思路:**


　　双指针，`i`指向最左边出现`0`的位置，`j`遍历`nums`，且`i < j`，每次交换`nums[i]`和`nums[j]`即可。  

#### **代码:**

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = 0
        for j, num in enumerate(nums):
            if num == 0:
                continue

            while i < j - 1 and nums[i] != 0:
                i += 1
            if i < j and nums[i] == 0:
                nums[i], nums[j] = nums[j], nums[i]  # 交换
                i += 1

```

## A424. 替换后的最长重复字符

难度`中等`

#### 题目描述

给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，总共可最多替换 *k* 次。在执行上述操作后，找到包含重复字母的最长子串的长度。

**注意:**
字符串长度 和 *k* 不会超过 104。

> **示例 1:**

```
输入:
s = "ABAB", k = 2

输出:
4

解释:
用两个'A'替换为两个'B',反之亦然。
```

> **示例 2:**

```
输入:
s = "AABABBA", k = 1

输出:
4

解释:
将中间的一个'A'替换为'B',字符串变为 "AABBBBA"。
子串 "BBBB" 有最长重复字母, 答案为 4。
```


#### 题目链接

<https://leetcode-cn.com/problems/longest-repeating-character-replacement/>

#### **思路:**

　　双指针，记录窗口内每个字母出现的次数。更新过程如下：  

　　① 保证窗口内`最多字母的出现次数`-`其他字母数的出现次数`<=`k`；  

　　② 如果窗口的大小大于之前的最大值，则更新最大值。  

#### **代码:**

```python
from collections import defaultdict

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        mem = defaultdict(int)  # 记录窗口内有哪些字母
        if not s:
            return 0
        left = 0
        ans = 1
        max_times = 0  # 最大出现次数
        for right, char in enumerate(s):
            mem[char] += 1
            max_times = max(max_times, mem[char])
						# right - left + 1 是窗口内的字母数
            while right - left + 1 - max_times > k:  # 其他的字母数大于k
                char_left = s[left]
                mem[char_left] -= 1
                left += 1

            ans = max(ans, right - left + 1)  # 以right结尾

        return ans

```

## A457. 环形数组循环

难度`中等`

#### 题目描述

给定一个含有正整数和负整数的**环形**数组 `nums`。 如果某个索引中的数 *k* 为正数，则向前移动 *k* 个索引。相反，如果是负数 (*-k*)，则向后移动 *k* 个索引。因为数组是环形的，所以可以假设最后一个元素的下一个元素是第一个元素，而第一个元素的前一个元素是最后一个元素。

确定 `nums` 中是否存在循环（或周期）。循环必须在相同的索引处开始和结束并且循环长度 > 1。此外，一个循环中的所有运动都必须沿着同一方向进行。换句话说，一个循环中不能同时包括向前的运动和向后的运动。

> **示例 1：**

```
输入：[2,-1,1,2,2]
输出：true
解释：存在循环，按索引 0 -> 2 -> 3 -> 0 。循环长度为 3 。
```

> **示例 2：**

```
输入：[-1,2]
输出：false
解释：按索引 1 -> 1 -> 1 ... 的运动无法构成循环，因为循环的长度为 1 。根据定义，循环的长度必须大于 1 。
```

> **示例 3:**

```
输入：[-2,1,-1,-2,-2]
输出：false
解释：按索引 1 -> 2 -> 1 -> ... 的运动无法构成循环，因为按索引 1 -> 2 的运动是向前的运动，而按索引 2 -> 1 的运动是向后的运动。一个循环中的所有运动都必须沿着同一方向进行。
```
**提示：**

1. -1000 ≤ nums[i] ≤ 1000
2. nums[i] ≠ 0
3. 0 ≤ nums.length ≤ 5000
**进阶：**

你能写出时间时间复杂度为 **O(n)** 和额外空间复杂度为 **O(1)** 的算法吗？

#### 题目链接

<https://leetcode-cn.com/problems/circular-array-loop/>

#### **思路:**

　　如果数组中存在环，也就是从**环的任意位置开始**向同一个方向循环，最终一定能回到环中。  

　　从数组的每个元素开始，不断向同一个方向往前走，看最终能不能回到走过的元素中。  

　　为了减少不必要的重复，用`visited`数组标记访问过的元素。  

#### **代码:**

```python
class Solution:
    def circularArrayLoop(self, nums: List[int]) -> bool:
        n = len(nums)
        visited = [False for _ in range(n)]
        for i, num in enumerate(nums):
            if visited[i]:
                continue
            j = i
            circle = set()  # 记录环中的元素
            while not visited[j]:
                circle.add(j)
                visited[j] = True
                nxt = (j + nums[j]) % n
                if nxt == j or nums[nxt] * nums[j] < 0:  # 不能是同一个元素且方向要为同向
                    break
                j = nxt
            else:
                if j in circle:
                    return True

        return False

```

## A532. 数组中的K-diff数对

难度`简单`

#### 题目描述

给定一个整数数组和一个整数 **k**, 你需要在数组里找到**不同的** k-diff 数对。这里将 **k-diff** 数对定义为一个整数对 (i, j), 其中 **i** 和 **j** 都是数组中的数字，且两数之差的绝对值是 **k**。

> **示例 1:**

```
输入: [3, 1, 4, 1, 5], k = 2
输出: 2
解释: 数组中有两个 2-diff 数对, (1, 3) 和 (3, 5)。
尽管数组中有两个1，但我们只应返回不同的数对的数量。
```

> **示例 2:**

```
输入:[1, 2, 3, 4, 5], k = 1
输出: 4
解释: 数组中有四个 1-diff 数对, (1, 2), (2, 3), (3, 4) 和 (4, 5)。
```

> **示例 3:**

```
输入: [1, 3, 1, 5, 4], k = 0
输出: 1
解释: 数组中只有一个 0-diff 数对，(1, 1)。
```

**注意:**

1. 数对 (i, j) 和数对 (j, i) 被算作同一数对。
2. 数组的长度不超过10,000。
3. 所有输入的整数的范围在 [-1e7, 1e7]。

#### 题目链接

<https://leetcode-cn.com/problems/k-diff-pairs-in-an-array/>

#### **思路:**

　　① `k < 0`时，直接返回`0`；  

　　②  `k = 0`时，返回`出现次数大于1`的数字个数；  

　　③ `k > 0`时，因为要找不同的`k-diff`数对，出现多次的数字并没有用。设所有出现数字的集合为`c`，如果`num`和`num + k`同时再`c`中出现，则结果数`+1`。  

#### **代码:**

```python
class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        c = set(nums)
        ans = 0
        if k < 0:
            return 0
        if k == 0:  # 特判k = 0时， 返回出现次数多于1次的数字个数  
            for num in c:
                if nums.count(num) > 1:
                    ans += 1
            return ans

        for num in c:  # 如果num和num+k都在集合中，结果数+1
            if num + k in c:
                ans += 1

        return ans

```

## A567. 字符串的排列

难度`中等`

#### 题目描述

给定两个字符串 **s1** 和 **s2**，写一个函数来判断 **s2** 是否包含 **s1** 的排列。

换句话说，第一个字符串的排列之一是第二个字符串的子串。

> **示例1:**

```
输入: s1 = "ab" s2 = "eidbaooo"
输出: True
解释: s2 包含 s1 的排列之一 ("ba").
```
> **示例2:**

```
输入: s1= "ab" s2 = "eidboaoo"
输出: False
```
**注意：**

1. 输入的字符串只包含小写字母
2. 两个字符串的长度都在 [1, 10,000] 之间

#### 题目链接

<https://leetcode-cn.com/problems/permutation-in-string/>

#### **思路:**

　　滑动窗口内计数。  

　　① 因为要判断`s1`的排列，所以窗口的大小是**固定的**(`len(s1)`)。  

　　② 输入的字符只包含小写字母，所以用一个数组对所有的小写字母计数。  

　　③ 出现次数为负表示窗口内该字符缺少几个，为正的字符是不需要的(这样更方便排除掉不需要的字符)。  

　　④ 如果当前窗口内所有字母出现次数都为`"0"`(表示没有缺少的字母)，则返回`True`。  　　

#### **代码:**

```python
from collections import defaultdict

class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        mem = defaultdict(int)
        for char in s1:
            mem[char] -= 1  # 把s1中的字母扣掉
        
        l1 = len(s1)
        lack_count = l1  # 当前窗口内缺少几个字母

        left = 0
        # 固定窗口长度
        for right, char in enumerate(s2):
            if mem[char] < 0:
                lack_count -= 1  # 原来缺少的字母出现了 因此缺少的字母数-1
            mem[char] += 1

            # 保证窗口的大小为 l1
            if right - left >= l1:
                mem[s2[left]] -= 1
                if mem[s2[left]] < 0:  # 缺数字
                    lack_count += 1
                left += 1

            if lack_count == 0:
                return True

        return False

```

## A632. 最小区间

难度`困难`

#### 题目描述

你有 `k` 个升序排列的整数数组。找到一个**最小**区间，使得 `k` 个列表中的每个列表至少有一个数包含在其中。

我们定义如果 `b-a < d-c` 或者在 `b-a == d-c` 时 `a < c`，则区间 [a,b] 比 [c,d] 小。

> **示例 1:**

```
输入:[[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]
输出: [20,24]
解释: 
列表 1：[4, 10, 15, 24, 26]，24 在区间 [20,24] 中。
列表 2：[0, 9, 12, 20]，20 在区间 [20,24] 中。
列表 3：[5, 18, 22, 30]，22 在区间 [20,24] 中。
```

**注意:**

1. 给定的列表可能包含重复元素，所以在这里升序表示 >= 。
2. 1 <= `k` <= 3500
3. -105 <= `元素的值` <= 105
4. **对于使用Java的用户，请注意传入类型已修改为List<List<Integer>>。重置代码模板后可以看到这项改动。**


#### 题目链接

<https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/>

#### **思路:**

　　**方法一：**双指针， `left`和`right`分别指向区间的左端点和右端点。维护一个长度**固定为k**的升序数组`asc`，每个列表中都放一个值，初始状态`asc`存放的是**每个列表的第一个元素**。步骤如下：  

　　① `asc`中最大的元素即为当前区间的右端点`right`，最小的元素即为区间的左端点`left`；  

　　② 每次都去除`asc`中**最小**的元素，然后到它所在的列表中找下一个元素，再将它插入到`asc`中，并且更新`left`和`right`；  

　　③ 如果当前区间大小比最小的区间还要小，则更新最小的区间；  

　　④ 当某个列表的元素用完时，结束循环。  

　　下面的动画表示`示例1`的查找过程：  

<embed src="_img/a632.mp4" width=700 height=408/>


　　**方法二：**最小堆，堆中存放的结构为`(列表中元素的值, 列表的下标, 该列表当前元素的索引)`。步骤如下：  

　　① 最小的元素出堆，然后到它所在的列表中找`下一个元素`，再将它按上面的结构插入到堆中，并且更新`left`和`right`；  

　　② `left`的值就是每次出堆的最小元素值，`right`的值初始是所有列表第一个元素中的最大值，之后如果小于出堆元素所在列表的`下一个元素`，则更新`right`；  

　　③、④和方法一的③、④相同。  

#### **代码:**

　　**方法一：**（双指针, 320ms）

```python
import bisect   

class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        k = len(nums)
        firsts = [num[0] for num in nums]  # 第一个元素
        left, right = min(firsts), max(firsts)
        min_left, min_right = left, right

        asc = sorted(list(zip(firsts, range(k))))  # 和下标对应

        indics = [0 for _ in range(k)]  # 每个数组的下标
        while True:
            peek, i = asc.pop(0)  # 删除最小的
            indics[i] += 1  # k个数组中的第i个，当前下标加1
            if indics[i] >= len(nums[i]):
                break
            idx = indics[i]
            bisect.insort(asc, (nums[i][idx], i))  # (数值, 数组的下标)
            
            right = asc[-1][0]
            left = asc[0][0]
            if right - left < min_right - min_left:  # 越小的越好 所以要严格小于
                min_right, min_left = right, left

        return [min_left, min_right]

```

　　**方法二：**（最小堆, 276ms）

```python
import bisect   

class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        import heapq

        heap = [(row[0], i, 0) for i, row in enumerate(nums)]
        heapq.heapify(heap)
        min_left, min_right = float('-inf'), float('inf')

        right = max(heap)[0]

        while heap:
            left, i, j = heapq.heappop(heap)  # i表示列表的下标, j表示列表中当前元素的索引
            if right - left < min_right - min_left:
                min_right, min_left = right, left

            if j + 1 >= len(nums[i]):  # 元素用完了
                return [min_left, min_right]

            nxt = nums[i][j+1]
            right = max(right, nxt)
            heapq.heappush(heap, (nxt, i, j + 1))

```

## A713. 乘积小于K的子数组

难度`中等`

#### 题目描述

给定一个正整数数组 `nums`。

找出该数组内乘积小于 `k` 的连续的子数组的个数。

> **示例 1:**

```
输入: nums = [10,5,2,6], k = 100
输出: 8
解释: 8个乘积小于100的子数组分别为: [10], [5], [2], [6], [10,5], [5,2], [2,6], [5,2,6]。
需要注意的是 [10,5,2] 并不是乘积小于100的子数组。
```

**说明:**

- `0 < nums.length <= 50000`
- `0 < nums[i] < 1000`
- `0 <= k < 10^6`


#### 题目链接

<https://leetcode-cn.com/problems/subarray-product-less-than-k/>

#### **思路:**


　　双指针，统计以每个数字结尾的乘积小于`k`的连续子数组的个数然后累加起来即可。  

#### **代码:**

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0

        left = 0
        wnd = 1  # 窗口内元素的乘积
        ans = 0
        for right, num in enumerate(nums):
            wnd *= num
            while wnd >= k:
                wnd = wnd // nums[left]
                left += 1
            ans += right - left + 1  # 以num结尾的满足条件的连续子数组个数

        return ans

```


## A763. 划分字母区间

难度`中等`

#### 题目描述

字符串 `S` 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一个字母只会出现在其中的一个片段。返回一个表示每个字符串片段的长度的列表。

> **示例 1:**

```
输入: S = "ababcbacadefegdehijhklij"
输出: [9,7,8]
解释:
划分结果为 "ababcbaca", "defegde", "hijhklij"。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
```

**注意:**

1. `S`的长度在`[1, 500]`之间。
2. `S`只包含小写字母`'a'`到`'z'`。


#### 题目链接

<https://leetcode-cn.com/problems/partition-labels/>

#### **思路:**


　　只需要统计每个字母第一次出现的下标（find函数），和最后一次出现的下标的（rfind函数），就可以得到每个字母的区间范围。  

　　比如示例1 的对应统计数组为： [[0, 8], [1, 5], [4, 7], [9, 14], [10, 15], [11, 11], [13, 13], [16, 19], [17, 22], [18, 23], [20, 20], [21, 21]]。

　　然后题目就会转化为[A56. 合并区间](/array?id=a56-合并区间)。  

#### **代码:**

```python
import string
from collections import defaultdict 

class Solution:

  	# A56. 合并区间
    def merge(self, intervals):
        if len(intervals) == 0:
            return []
        s = sorted(intervals)
        ans = [s[0]]
        for i in s[1:]:
            if i[0] <= ans[-1][1]:
                ans[-1] = [ans[-1][0], max(i[1], ans[-1][1])] 
            else:
                ans.append(i)

        return ans

    def partitionLabels(self, S: str) -> List[int]:
        mem = defaultdict(tuple)
        for char in string.ascii_lowercase:  # 查找所有的小写字母
            if S.find(char) >= 0:
                mem[char] = (S.find(char), S.rfind(char))

        intervals = list(mem.values())
        merge = self.merge(intervals)
        return [j - i + 1 for i, j in merge]

```
