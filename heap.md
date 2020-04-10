# 堆

## A218. 天际线问题

难度`困难`

#### 题目描述

城市的天际线是从远处观看该城市中所有建筑物形成的轮廓的外部轮廓。现在，假设您获得了城市风光照片（图A）上**显示的所有建筑物的位置和高度**，请编写一个程序以输出由这些建筑物**形成的天际线**（图B）。

<img src="_img/218_1.png" style="zoom:40%"/> <<img src="_img/218_2.png" style="zoom:40%"/> >

每个建筑物的几何信息用三元组 `[Li，Ri，Hi]` 表示，其中 `Li` 和 `Ri` 分别是第 i 座建筑物左右边缘的 x 坐标，`Hi` 是其高度。可以保证 `0 ≤ Li, Ri ≤ INT_MAX`, `0 < Hi ≤ INT_MAX` 和 `Ri - Li > 0`。您可以假设所有建筑物都是在绝对平坦且高度为 0 的表面上的完美矩形。

例如，图A中所有建筑物的尺寸记录为：`[ [2 9 10], [3 7 15], [5 12 12], [15 20 10], [19 24 8] ] `。

输出是以 `[ [x1,y1], [x2, y2], [x3, y3], ... ]` 格式的“**关键点**”（图B中的红点）的列表，它们唯一地定义了天际线。**关键点是水平线段的左端点**。请注意，最右侧建筑物的最后一个关键点仅用于标记天际线的终点，并始终为零高度。此外，任何两个相邻建筑物之间的地面都应被视为天际线轮廓的一部分。

例如，图B中的天际线应该表示为：`[ [2 10], [3 15], [7 12], [12 0], [15 10], [20 8], [24, 0] ]`。

**说明:**

- 任何输入列表中的建筑物数量保证在 `[0, 10000]` 范围内。
- 输入列表已经按左 `x` 坐标 `Li`  进行升序排列。
- 输出列表必须按 x 位排序。
- 输出天际线中不得有连续的相同高度的水平线。例如 `[...[2 3], [4 5], [7 5], [11 5], [12 7]...]` 是不正确的答案；三条高度为 5 的线应该在最终输出中合并为一个：`[...[2 3], [4 5], [12 7], ...]`

#### 题目链接

<https://leetcode-cn.com/problems/the-skyline-problem/>

#### **思路:**

　　用一个`大根堆`存放当前所有建筑的高度，在建筑的边界时，把堆顶的元素放入结果数组中。  

　　注意不能输出连续相同高度的水平线，所以要判断结果数组最后一个元素是不是和堆顶的元素相同。  

　　例如，示例中的过程为：  

```python
x = 2, 蓝色房子入堆, heap = [10], ans = [[2 10]]
x = 3, 红色房子入堆, heap = [10, 15], ans = [[2 10], [3 15]]
x = 5, 绿色房子入堆, heap = [10, 12, 15]，堆顶元素高度为15，但是ans最后已经出现15了，忽略
x = 7, 最高的房子(红色房子)出堆, heap = [10, 12]，堆顶高度12, ans = [[2 10], [3 15], [7 12]]
x = 9, 最高的房子(绿色房子)还未结束, 不出堆, 堆顶高度12, 与ans[-1]重复，忽略
x = 12, 绿色房子和蓝色房子都已经结束, 均出堆, 堆中已没有元素, 插入0, ans = [[2 10], [3 15], [7 12], [12 0]]
x = 15, 紫色房子入堆, heap = [10], ans = [[2 10], [3 15], [7 12], [12 0], [15 10]]
x = 19, 黄色房子入堆, heap = [10, 8], 最高的房子还未结束, 不出堆, 堆顶高度10, 重复，忽略, ans不变
x = 20, 紫色房子出堆, heap = [8], 堆顶高度8, ans = [[2 10], [3 15], [7 12], [12 0], [15 10], [20 8]]
x = 24, 黄色房子出堆, 堆中已没有元素, 插入0, ans = [[2 10], [3 15], [7 12], [12 0], [15 10], [20 8], [24 0]]

```

#### **代码:**

```python
import heapq
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        n = len(buildings)
        x = [b[0] for b in buildings] + [b[1] for b in buildings]
        x.sort()

        heap = []
        ans = [[0, 0]]
        idx = 0

        for i in x:
            while idx < n and buildings[idx][0] == i:
                h, right = buildings[idx][2], buildings[idx][1]
                heapq.heappush(heap, (-h, right))  # 大根堆
                idx += 1

            while heap:
                h, right = heapq.heappop(heap)
                if right > i:  # 还没有结束
                    heapq.heappush(heap, (h, right))  # 再放回去
                    if ans[-1][1] != -h:
                        ans.append([i, -h])
                    break
            else:
                if ans[-1][1] != 0:
                    ans.append([i, 0])

        return ans[1:]

```

## A264. 丑数 II

难度 `中等`  

#### 题目描述

编写一个程序，找出第 `n` 个丑数。

丑数就是只包含质因数 `2, 3, 5` 的**正整数**。

> **示例:**

```
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
```

**说明:**  

1. `1` 是丑数。
2. `n` **不超过**1690。

#### 题目链接

<https://leetcode-cn.com/problems/ugly-number-ii/>

#### 思路  

　　方法一：小顶堆，每次取堆顶的元素（也就是最小的），第`i`次取的就是第`i`个丑数。再把它分别乘以`2`、`3`、`5`插入到堆中，如下图所示：  

　　<img src="_img/a264_1.png" style="zoom:33%"/>  

　　为了避免出现重复，用一个集合`used_set`记录已经出现过的元素。已经出现过的元素就不会再入堆了。  

　　方法二：三指针法。使用三个指针`id_2`、`id_3`、`id_5`，分别表示2、3、5应该乘以丑数数组中的哪个元素。如下图所示：  

　　<img src="_img/a264_2.png" style="zoom:40%"/>

　　初始时丑数数组 =`[1]`，三个指针均为`0`，比较三个指针乘积的结果，把最小的作为下一个丑数，并且这个指针向右移`1`。  

　　如果有多个指针乘积结果相同（如图中的2×3=3×2），则同时移动它们。  

#### 代码  

方法一（小顶堆+去重）：

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        if n == 1: return 1
        cur = 1
        used_set = {1}
        factors = (2, 3, 5)
        
        import heapq
        heap = []
        for i in range(2, n+1):
            for f in factors:
                new_num = cur * f
                if new_num not in used_set:
                    used_set.add(new_num)
                    heapq.heappush(heap, new_num)
            cur = heapq.heappop(heap)

        return cur

```

方法二（三指针）：

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        if n == 1: return 1
        result = [0] * n
        result[0] = 1
        id_2, id_3, id_5 = 0, 0, 0
        for i in range(1, n):
            a = result[id_2] * 2
            b = result[id_3] * 3
            c = result[id_5] * 5
            minimal = min(a, b, c)

            if a == minimal: id_2 += 1
            if b == minimal: id_3 += 1
            if c == minimal: id_5 += 1

            result[i] = minimal
            
        return result[-1]
```

