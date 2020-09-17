# 贪心算法

## A45. 跳跃游戏 II 

难度 `困难`

#### 题目描述

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

> **示例:**

```
输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```

> **说明:**

假设你总是可以到达数组的最后一个位置。

#### 题目链接

<https://leetcode-cn.com/problems/jump-game-ii/>

#### 思路  

　　贪心算法，每次都跳到最划算的位置。`数值大的位置`会更加划算，`距离当前位置更远的`也会更加划算。  

　　设下一个位置与当前位置`i`的距离为`j`，即优化`nums[i + j] + j`最大即可找到下一个位置。  
　　例如`[2, 3, 1, 1, 4]`。初始`i = 0`，`nums[i] = 2`，能够跳到的两个位置中，`3`的位置偏差为`1`，`1`的位置偏差为`2`；而`3+1 > 1+2`。因此跳到`3`的位置更为划算。  

#### 代码  

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
        i = 0  # 当前位置
        max_indic = 0  # 记录跳到最划算位置的下标
        ans = 1
        while i + nums[i] < n - 1:
            max_temp = 0
            num = nums[i]
            for j in range(1, num + 1):  # 这里的j表示跳到的位置和i的偏差
                if nums[i + j] + j > max_temp:
                    max_temp = nums[i + j] + j
                    max_indic = i + j
            ans += 1
            i = max_indic

        return ans
      
```

## A55. 跳跃游戏 

难度 `中等`

#### 题目描述

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个位置。

> **示例 1:**

```
输入: [2,3,1,1,4]
输出: true
解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。
```

> **示例 2:**

```
输入: [3,2,1,0,4]
输出: false
解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。
```

#### 题目链接

<https://leetcode-cn.com/problems/jump-game/>

#### 思路  

　　方法一：贪心算法。与[A45. 跳跃游戏II](/array?id=a45-跳跃游戏-ii)类似，每次都跳到最划算的位置。  
　　方法二：从右往左遍历，如果某个位置能走到最后则截断后面的元素。如果某个元素为`0`则从前面找能走到它后面的。方法二比方法一用时短一些。  

#### 代码  

　　方法一：

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True

        i = 0  # 当前位置
        while nums[i] != 0 and i < n-1:
            temp_indic = 0
            temp_max = -1
            for j in range(nums[i]):
                if i + j + 1 >= n - 1:
                    return True
                if nums[i + j + 1] + j > temp_max:
                    temp_indic = i + j + 1
                    temp_max = nums[i + j + 1] + j
            i = temp_indic

        return i >= n-1
```

　　方法二：

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True

        j = 0
        for i in range(n-2,-1,-1):
            if nums[i] == 0 or j > 0:  # 出现了或之前出现过0，则每次都加一
                j += 1
            if nums[i] >= j:  # 如果当前位置能跳过最后一个0，则归0
                j = 0

        return j == 0
```

## A122. 买卖股票的最佳时机 II

难度`简单`

#### 题目描述

给定一个数组，它的第 *i* 个元素是一支给定股票第 *i* 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

**注意：**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
> **示例 1:**

```
输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
```

> **示例 2:**

```
输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
```

> **示例 3:**

```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```
**提示：**

- `1 <= prices.length <= 3 * 10 ^ 4`
- `0 <= prices[i] <= 10 ^ 4`


#### 题目链接

<https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/>

#### **思路:**


　　贪心，如果某天的股票价格比前一天高，那么就赚取这两天之间的差价。  

#### **代码:**

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        ans = 0

        for i in range(1, n):
            if prices[i] > prices[i-1]:
                ans += prices[i] - prices[i-1]

        return ans

```

 ## A134. 加油站

难度`中等`

#### 题目描述

  在一条环路上有 *N* 个加油站，其中第 *i* 个加油站有汽油 `gas[i]` 升。

  你有一辆油箱容量无限的的汽车，从第 *i* 个加油站开往第 *i+1* 个加油站需要消耗汽油 `cost[i]` 升。你从其中的一个加油站出发，开始时油箱为空。

  如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

  **说明:** 

  - 如果题目有解，该答案即为唯一答案。
  - 输入数组均为非空数组，且长度相同。
  - 输入数组中的元素均为非负数。

>  **示例 1:**

  ```
  输入: 
  gas  = [1,2,3,4,5]
  cost = [3,4,5,1,2]
  
  输出: 3
  
  解释:
  从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
  开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
  开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
  开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
  开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
  开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
  因此，3 可为起始索引。
  ```

>  **示例 2:**

  ```
  输入: 
  gas  = [2,3,4]
  cost = [3,4,3]
  
  输出: -1
  
  解释:
  你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
  我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油
  开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油
  开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油
  你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。
  因此，无论怎样，你都不可能绕环路行驶一周。
  ```


#### 题目链接

<https://leetcode-cn.com/problems/gas-station/>

#### **思路:**


　　贪心算法。假设从编号为0站开始，一直到`k`站都正常，在开往`k+1`站时车子没油了。这时，应该将起点设置为`k+1`站。  

#### **代码:**

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        n = len(gas)
        nums = [gas[i] - cost[i] for i in range(n)]  # 油量净增长

        fuel = nums[0]

        ans = 0
        for i in range(1, n):
            if fuel < 0:  # 没油了
                ans = i
                fuel = nums[i]  # i设为起点，并且在i加油，重新开
            else:
                fuel += nums[i]

        return ans

```

## A135. 分发糖果

难度`困难`

#### 题目描述

老师想给孩子们分发糖果，有 *N* 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

- 每个孩子至少分配到 1 个糖果。
- 相邻的孩子中，评分高的孩子必须获得更多的糖果。

那么这样下来，老师至少需要准备多少颗糖果呢？

> **示例 1:**

```
输入: [1,0,2]
输出: 5
解释: 你可以分别给这三个孩子分发 2、1、2 颗糖果。
```

> **示例 2:**

```
输入: [1,2,2]
输出: 4
解释: 你可以分别给这三个孩子分发 1、2、1 颗糖果。
     第三个孩子只得到 1 颗糖果，这已满足上述两个条件。
```


#### 题目链接

<https://leetcode-cn.com/problems/candy/>

#### **思路:**


　　按评分排序，评分最少的给1个糖，然后从评分低到高依次访问。如果有前后的小朋友分数低但糖更多的(假设有`n`个糖)，就给这个小朋友分`n+1`个糖。

#### **代码:**

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        candies = [0 for _ in range(n)]

        mem = sorted([(r, i) for i, r in enumerate(ratings)])
        
        for _, i in mem:  # 按评分从低到高访问
            if i >= 1 and ratings[i] > ratings[i-1]:  # 比前面的更高
                if candies[i] <= candies[i-1]:  # 糖却少
                    candies[i] = candies[i-1] + 1

            if i < n - 1 and ratings[i] > ratings[i+1]:  # 比后面的更高
                if candies[i] <= candies[i+1]:  # 糖却少
                    candies[i] = candies[i+1] + 1

        return sum(candies) + n

```

## A316. 去除重复字母

难度`困难`

#### 题目描述

给你一个仅包含小写字母的字符串，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证返回结果的字典序最小（要求不能打乱其他字符的相对位置）。

> **示例 1:**

```
输入: "bcabc"
输出: "abc"
```

> **示例 2:**

```
输入: "cbacdcbc"
输出: "acdb"
```

#### 题目链接

<https://leetcode-cn.com/problems/remove-duplicate-letters/>

#### **思路:**

　　**方法一：**递归，假设`s`中最小的字母是`"a"`，先看第一个`"a"`右边是不是所有其他的字母都有，如果都有，把右边的部分中的`"a"`去掉，然后递归地调用本算法。  

　　如果右边不是所有的字母都有，则继续查找第二小的字母。  

　　例如示例2的递归过程如下：  

```python
① 最小的是"a"，"a"的右边所有字母都有, ans = "a"，递归处理"cdcbc"；
② 最小的是"b"，"b"的右边没有"d"，继续找下一个；
    下一个最小的是"c"，"c"的右边所有字母都有，ans = "ac"， 递归处理"db"；
③ 最小的是"b"，"b"的右边没有"d"，继续找下一个；
    下一个最小的是"d"，"b"的右边所有字母都有，ans = "acd"， 递归处理"b"；
④ 只剩一个字母了，返回"b"，ans = "acdb"；

```

　　**方法二：**单调栈，思路就是，遇到一个新字符，如果比`栈顶`小，并且在后面和`栈顶`一样的字母还有，就把栈顶的字符抛弃了。  

#### **代码:**

　　**方法一：**递归(68ms)

```python
import string
class Solution:        
    def removeDuplicateLetters(self, s: str) -> str:
        all_letters = set(s)
        minimal_letters = sorted(list(all_letters))

        # 先找a 看a右边是不是所有字母都有，再找b
        for char in minimal_letters:
            idx = s.find(char)
            if set(s[idx:]) == all_letters:
                return char + self.removeDuplicateLetters(s[idx:].replace(char, ''))

        return ''

```

　　**方法二：**单调栈(44ms)

```python
import string
class Solution:        
    def removeDuplicateLetters(self, s: str) -> str:
        used = set()
        
        stack = ['#']  # 井号的ascii码小于所有小写字母

        ans = ''
        for i, char in enumerate(s):
            if char not in used:
                while char < stack[-1] and s.find(stack[-1], i+1) != -1 :  # 比栈顶小
                    top = stack.pop()
                    used.remove(top)

                stack.append(char)
                used.add(char)

        return ''.join(stack[1:])  # 最后将栈中的元素按顺序输出就是答案
                
```

## A330. 按要求补齐数组

难度`困难`

#### 题目描述

给定一个已排序的正整数数组 *nums，* 和一个正整数 *n 。* 从 `[1, n]` 区间内选取任意个数字补充到 *nums* 中，使得 `[1, n]` 区间内的任何数字都可以用 *nums* 中某几个数字的和来表示。请输出满足上述要求的最少需要补充的数字个数。

> **示例 1:**

```
输入: nums = [1,3], n = 6
输出: 1 
解释:
根据 nums 里现有的组合 [1], [3], [1,3]，可以得出 1, 3, 4。
现在如果我们将 2 添加到 nums 中， 组合变为: [1], [2], [3], [1,3], [2,3], [1,2,3]。
其和可以表示数字 1, 2, 3, 4, 5, 6，能够覆盖 [1, 6] 区间里所有的数。
所以我们最少需要添加一个数字。
```

> **示例 2:**

```
输入: nums = [1,5,10], n = 20
输出: 2
解释: 我们需要添加 [2, 4]。
```

> **示例 3:**

```
输入: nums = [1,2,2], n = 5
输出: 0
```


#### 题目链接

<https://leetcode-cn.com/problems/patching-array/>

#### **思路:**

　　**方法一：**记录所有求和可以得到的数字，遍历1~n，如果某个数字无法得到，就补上这个数字，并重新计算所有可以得到的数字。(超时)  　　

　　**方法二：**用一个变量`fit_max`表示当前求和可以得到的最大数字，( `[1, fit_max]`之间的所有数字都可以得到)。  

　　① 如果`nums`中有**未使用过**的数字`num`，并且它小于等于`fit_max + 1`，那么可以用上它，`fit_max = fit_max + num`；  

　　② 如果`nums`中的数字都已经用过了，或者没用过的均大于`fit_max + 1`，那么只能补上`fit_max + 1`。之后`fit_max` 更新为`2 * fit_max + 1`。

#### **代码:**

　　**方法一：**遍历1~n(超时)

```python
class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:

        set_t = set()
        
        def add(num):
            if set_t:
                new_added = set()
                for t in set_t:
                    new_added.add(num + t)
                set_t.update(new_added)

            set_t.add(num)

        for num in nums:
            add(num)

        ans = 0 
        for i in range(1, n + 1):
            if i not in set_t:
                ans += 1
                add(i)

        return ans
```

　　**方法二：**记录最大能得到的数(80ms)

```python
class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        idx = 0  # nums中的id
        ln = len(nums)

        fit_max = 0  # 当前能够表示的最大范围为[1, fit_max]
        ans = 0
        while fit_max < n:
            if idx < ln and nums[idx] <= fit_max + 1:  # 如果 nums 中有小于等于 fit_max + 1的数字，可以直接用上
                fit_max = nums[idx] + fit_max
                idx += 1
            else:  # 无法用nums中的数字
                # 添加fit_max + 1
                ans += 1
                fit_max = fit_max + fit_max + 1

        return ans

```

## A402. 移掉K位数字

难度`中等`

#### 题目描述

给定一个以字符串表示的非负整数 *num* ，移除这个数中的 *k* 位数字，使得剩下的数字最小。

**注意:**

- *num* 的长度小于 10002 且 ≥ *k。*
- *num* 不会包含任何前导零。

> **示例 1 :**

```
输入: num = "1432219", k = 3
输出: "1219"
解释: 移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219。
```

> **示例 2 :**

```
输入: num = "10200", k = 1
输出: "200"
解释: 移掉首位的 1 剩下的数字为 200. 注意输出不能有任何前导零。
```

示例 **3 :**

```
输入: num = "10", k = 2
输出: "0"
解释: 从原数字移除所有的数字，剩余为空就是0。
```


#### 题目链接

<https://leetcode-cn.com/problems/remove-k-digits/>

#### **思路:**


　　找到`num`的前`k+1`位中最小的，把它留下来作为结果的首位。将它前面的数字(假设有`a`个)**全部删除**；对它后面的数字继续计算移除`k-a`个数字能够得到的最小的数，递归地求解即可。  

#### **代码:**

```python
class Solution:
    def helper(self, num, k):
        if k == 0:
            return num

        ln = len(num)
        if ln == k:
            return ''

        first = min(num[:k+1])  # 结果最小的首位数字
        a = num.index(first)
        return first + self.helper(num[a+1:], k - a)

    def removeKdigits(self, num: str, k: int) -> str:
        ans = self.helper(num, k)

        ans = ans.lstrip('0')  # 去除前导0
        if not ans:
            ans = '0'
        return ans

```

## A406. 根据身高重建队列

难度`中等`

#### 题目描述

假设有打乱顺序的一群人站成一个队列。 每个人由一个整数对`(h, k)`表示，其中`h`是这个人的身高，`k`是排在这个人前面且身高大于或等于`h`的人数。 编写一个算法来重建这个队列。

**注意：**
总人数少于1100人。

> **示例**

```
输入:
[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]

输出:
[[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
```


#### 题目链接

<https://leetcode-cn.com/problems/queue-reconstruction-by-height/>

#### **思路:**

　　先按照`(k, h)`排序，然后用插入排序的思想。  

　　一个人如果身高为`h`，前面有`k`个身高大于等于他的人，那么插入排序时让他**尽量往后站**，只要保证前面身高大于等于他的人数为`k`即可。  

#### **代码:**

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key = lambda kv: (kv[1], kv[0]))
        # print(people)

        ans = []
        for h, k in people:
            count = 0
            i = 0
            for i, (h0, _) in enumerate(ans):
                if h0 >= h:
                    count += 1
                if count > k:
                    ans.insert(i, [h, k])
                    break
            else:
                ans.insert(i + 1, [h, k])

        return ans

```

## A435. 无重叠区间

难度`中等`

#### 题目描述

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

**注意:**

1. 可以认为区间的终点总是大于它的起点。
2. 区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。

> **示例 1:**

```
输入: [ [1,2], [2,3], [3,4], [1,3] ]

输出: 1

解释: 移除 [1,3] 后，剩下的区间没有重叠。
```

> **示例 2:**

```
输入: [ [1,2], [1,2], [1,2] ]

输出: 2

解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。
```

> **示例 3:**

```
输入: [ [1,2], [2,3] ]

输出: 0

解释: 你不需要移除任何区间，因为它们已经是无重叠的了。
```


#### 题目链接

<https://leetcode-cn.com/problems/non-overlapping-intervals/>

#### **思路:**

　　类似于[A56. 合并区间](/array?id=a56-合并区间)的思路。  

　　将所有区间排序，然后两两合并(取交集)，最后剩余的区间都是不重叠的。返回`原区间数`-`剩余区间数`。  

#### **代码:**

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals) == 0:
            return 0
        s = sorted(intervals)
        ans = [s[0]]

        for i in s[1:]:
            if i[0] < ans[-1][1]:
                ans[-1] = [max(ans[-1][0], i[0]), min(i[1], ans[-1][1])] 
            else:
                ans.append(i)

        return len(intervals) - len(ans)

```

## A452. 用最少数量的箭引爆气球

难度`中等`

#### 题目描述

在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以y坐标并不重要，因此只要知道开始和结束的x坐标就足够了。开始坐标总是小于结束坐标。平面内最多存在104个气球。

一支弓箭可以沿着x轴从不同点完全垂直地射出。在坐标x处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。

> **示例:**

```
输入:
[[10,16], [2,8], [1,6], [7,12]]

输出:
2

解释:
对于该样例，我们可以在x = 6（射爆[2,8],[1,6]两个气球）和 x = 11（射爆另外两个气球）。
```


#### 题目链接

<https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/>

#### **思路:**

　　可以参考[A56. 合并区间](/array?id=a56-合并区间)和[A435. 无重叠区间](/greedy?id=a435-无重叠区间)的思路。  

　　将所有的气球🎈排序，然后两两合并，最后返回区间的数量，注意`A56. 合并区间`是取并集，这一题是取交集。  　

#### **代码:**

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:

        if len(points) == 0:
            return 0
        s = sorted(points)
        ans = [s[0]]
        # print(s)

        for i in s[1:]:
            if i[0] <= ans[-1][1]:
                ans[-1] = [max(ans[-1][0], i[0]), min(i[1], ans[-1][1])] 
            else:
                ans.append(i)

        return len(ans)

```

## A455. 分发饼干

难度`简单`

#### 题目描述

假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。对每个孩子 i ，都有一个胃口值 gi ，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j ，都有一个尺寸 sj 。如果 sj >= gi ，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

**注意：**

你可以假设胃口值为正。
一个小朋友最多只能拥有一块饼干。

> **示例 1:**

```
输入: [1,2,3], [1,1]

输出: 1

解释: 
你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
所以你应该输出1。
```

> **示例 2:**

```
输入: [1,2], [1,2,3]

输出: 2

解释: 
你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
你拥有的饼干数量和尺寸都足以让所有孩子满足。
所以你应该输出2.
```


#### 题目链接

<https://leetcode-cn.com/problems/assign-cookies/>

#### **思路:**

　　要满足尽量多的孩子， 只能用尽量小的饼干🍪先发放给胃口小的。  

　　先排序，然后用最小的饼干尝试给胃口最小的孩子：① 如果不够，换下一块饼干 ② 如果够了，换下一个孩子。  

#### **代码:**

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        i, j = 0, 0
        ans = 0

        while i < len(g) and j < len(s):
            if s[j] >= g[i]:
                i += 1  # 下一个孩子
                j += 1
                ans += 1
            else:
                j += 1  # 下一块饼干

        return ans

```

## A621. 任务调度器

难度`中等`

#### 题目描述

给定一个用字符数组表示的 CPU 需要执行的任务列表。其中包含使用大写的 A - Z 字母表示的26 种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。CPU 在任何一个单位时间内都可以执行一个任务，或者在待命状态。

然而，两个**相同种类**的任务之间必须有长度为 **n** 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。

你需要计算完成所有任务所需要的**最短时间**。

> **示例 ：**

```
输入：tasks = ["A","A","A","B","B","B"], n = 2
输出：8
解释：A -> B -> (待命) -> A -> B -> (待命) -> A -> B.
```

**提示：**

1. 任务的总个数为 `[1, 10000]`。
2. `n` 的取值范围为 `[0, 100]`。

#### 题目链接

<https://leetcode-cn.com/problems/task-scheduler/>

#### **思路:**

　　**方法一：**模拟所有任务的执行，每次都执行当前能执行的数量最多的任务。  

　　**方法二：** 只考虑最多的任务情况，假设最多的任务出现了`x`次，则这个任务必须有`(x-1) * (n+1)`的间隔时间；如果最多的任务同时有`a`项，那么总的时间为`(x-1) * (n+1) + a`。  

　　除了最多的任务以外，其他的任务不用担心冷却的问题，但是可能数量过多而需要额外的时间，因此最终需要的时间为`max((x-1) * (n+1) + a, len(tasks))`。  

#### **代码:**

　　**方法一：**模拟所有任务的执行(1676ms)

```python
from collections import Counter
import bisect

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        # 每次都执行最多的任务
        c = Counter(tasks)
        asc = []
        ans = 0
        lasts = {}  # 最后一次执行的时间
        for k in c:
            bisect.insort(asc, (c[k], k))
            lasts[k] = -999

        while asc:
            ans += 1
            for i in range(len(asc)-1,-1,-1):  # 从后往前
                times, char = asc[i]
                if ans - lasts[char] > n:
                    asc.pop(i)
                    times -= 1
                    if times:
                        bisect.insort(asc, (times, char))
                    lasts[char] = ans
                    break
        
        return ans

```

　　**方法二：**只考虑次数最多的任务(48ms)

```python
from collections import Counter
import bisect

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        c = Counter(tasks)
        ans = 0
        x = max(c.values())
        ans = (x - 1) * (n + 1)

        ans += list(c.values()).count(x)
        
        return max(ans, len(tasks))
        
```

## A630. 课程表 III

难度`困难`

#### 题目描述

这里有 `n` 门不同的在线课程，他们按从 `1` 到 `n` 编号。每一门课程有一定的持续上课时间（课程时间）`t` 以及关闭时间第 d 天。一门课要持续学习 `t` 天直到第 d 天时要完成，你将会从第 1 天开始。

给出 `n` 个在线课程用 `(t, d)` 对表示。你的任务是找出最多可以修几门课。

> **示例：**

```
输入: [[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]
输出: 3
解释: 
这里一共有 4 门课程, 但是你最多可以修 3 门:
首先, 修第一门课时, 它要耗费 100 天，你会在第 100 天完成, 在第 101 天准备下门课。
第二, 修第三门课时, 它会耗费 1000 天，所以你将在第 1100 天的时候完成它, 以及在第 1101 天开始准备下门课程。
第三, 修第二门课时, 它会耗时 200 天，所以你将会在第 1300 天时完成它。
第四门课现在不能修，因为你将会在第 3300 天完成它，这已经超出了关闭日期。
```

**提示:**

1. 整数 1 <= d, t, n <= 10,000 。
2. 你不能同时修两门课程。

#### 题目链接

<https://leetcode-cn.com/problems/course-schedule-iii/>

#### **思路:**

　　贪心算法。首先解释一下这题为什么能用贪心算法：    

　　**性质一：**如果一门课程`b`的结束时间比`a`晚，而学习时间比`a`短，那么一定可以用课程`b`**替换掉**课程`a`课程。  

　　基于以上前提，我们先把所有课程按照**结束时间**排序。然后依次学习每门课程，如果某门课程无法在结束前学完，可以尝试替换掉之前耗时最长的课程。如果替换成功，可以把当前时间提前到`time = time + cost - longest_cost `(这样就留出更过的时间学习新的课程了)。  

　　具体实现上，可以用升序数组(`bisect`库)，也可以用大顶堆(`heapq`库)。  

#### **代码:**

　　**实现一：**bisect

```python
import bisect

class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        courses.sort(key= lambda kv:kv[1])
        chosen = []

        ans = 0
        time = 0
        for i in range(len(courses)):
            cost, end = courses[i]
            if time + cost <= end:
                bisect.insort(chosen, cost)
                time += cost
                ans += 1
            elif chosen:
                longest_cost = chosen[-1] 
                if cost < longest_cost:
                    chosen.pop()
                    bisect.insort(chosen, cost)
                    time = time + cost - longest_cost

        return ans

```

　　**实现二：**heap

```python
class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        courses.sort(key= lambda kv:kv[1])
        chosen = []

        ans = 0
        time = 0
        for i in range(len(courses)):
            cost, end = courses[i]
            if time + cost <= end:
                heapq.heappush(chosen, -cost)
                time += cost
                ans += 1
            elif chosen:
                longest_cost = - chosen[0] 
                if cost < longest_cost:
                    heapq.heappop(chosen)
                    heapq.heappush(chosen, -cost)
                    time = time + cost - longest_cost

        return ans

```

## A1578. 避免重复字母的最小删除成本

难度`中等`

#### 题目描述

给你一个字符串 `s` 和一个整数数组 `cost` ，其中 `cost[i]` 是从 `s` 中删除字符 `i` 的代价。

返回使字符串任意相邻两个字母不相同的最小删除成本。

请注意，删除一个字符后，删除其他字符的成本不会改变。

> **示例 1：**

```
输入：s = "abaac", cost = [1,2,3,4,5]
输出：3
解释：删除字母 "a" 的成本为 3，然后得到 "abac"（字符串中相邻两个字母不相同）。
```

> **示例 2：**

```
输入：s = "abc", cost = [1,2,3]
输出：0
解释：无需删除任何字母，因为字符串中不存在相邻两个字母相同的情况。
```

> **示例 3：**

```
输入：s = "aabaa", cost = [1,2,3,4,1]
输出：2
解释：删除第一个和最后一个字母，得到字符串 ("aba") 。
```

**提示：**

- `s.length == cost.length`
- `1 <= s.length, cost.length <= 10^5`
- `1 <= cost[i] <= 10^4`
- `s` 中只含有小写英文字母

#### 题目链接

<https://leetcode-cn.com/problems/minimum-deletion-cost-to-avoid-repeating-letters/>

#### **思路:**

　　有贪心算法的思想，如果相邻的两个字母相同，就删除花费小的那个。  

　　初始时维护一个空的数组`a`，将`s`中的字母逐个添加到`a`中，如果`s`中第`i`个字母与`a`最后一个字母相同，比较他们的`cost`，将`cost`小的删除，`cost`大的保留再`a`的末尾。  

#### **代码:**

**方法一：**模拟删除的过程（会超时）

```python
class Solution:
    def minCost(self, s: str, cost: List[int]) -> int:
        a = list(s)
        
        def helper(a, cost):
            for i in range(len(a)-1):
                if a[i] == a[i+1]:  # 要么删i 要么删i+1
                    if cost[i] <= cost[i+1]:
                        a.pop(i)
                        ans = cost.pop(i)
                    else:
                        a.pop(i+1)
                        ans = cost.pop(i+1)
                        
                    return ans + helper(a, cost)
                
            return 0
        
        return helper(a, cost)
      
```

**方法二：**

```python
class Solution:
    def minCost(self, s: str, cost: List[int]) -> int:
        a = list(s)
  
        temp = []
        costtemp = []
        ans = 0
        for i in range(len(a)):  # 前i个
            if not temp or temp[-1] != a[i]:
                temp.append(a[i])
                costtemp.append(cost[i])
            else:
                if cost[i] <= costtemp[-1]:  # 不入
                    ans += cost[i]
                else:
                    ans += costtemp[-1]
                    costtemp.pop()
                    costtemp.append(cost[i])
                
        return ans

```

