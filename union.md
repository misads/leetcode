# 并查集

## A1579. 保证图可完全遍历

难度`困难`

#### 题目描述

Alice 和 Bob 共有一个无向图，其中包含 n 个节点和 3  种类型的边：

- 类型 1：只能由 Alice 遍历。
- 类型 2：只能由 Bob 遍历。
- 类型 3：Alice 和 Bob 都可以遍历。

给你一个数组 `edges` ，其中 `edges[i] = [typei, ui, vi]` 表示节点 `ui` 和 `vi` 之间存在类型为 `typei` 的双向边。请你在保证图仍能够被 Alice和 Bob 完全遍历的前提下，找出可以删除的最大边数。如果从任何节点开始，Alice 和 Bob 都可以到达所有其他节点，则认为图是可以完全遍历的。

返回可以删除的最大边数，如果 Alice 和 Bob 无法完全遍历图，则返回 -1 。
> **示例 1：**

<img src="_img/5510_1.png" style="zoom:100%"/>

```
输入：n = 4, edges = [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,2],[2,3,4]]
输出：2
解释：如果删除 [1,1,2] 和 [1,1,3] 这两条边，Alice 和 Bob 仍然可以完全遍历这个图。再删除任何其他的边都无法保证图可以完全遍历。所以可以删除的最大边数是 2 。
```

> **示例 2：**

<img src="_img/5510_2.png" style="zoom:100%"/>

```
输入：n = 4, edges = [[3,1,2],[3,2,3],[1,1,4],[2,1,4]]
输出：0
解释：注意，删除任何一条边都会使 Alice 和 Bob 无法完全遍历这个图。
```

> **示例 3：**

<img src="_img/5510_3.png" style="zoom:100%"/>

```
输入：n = 4, edges = [[3,2,3],[1,1,2],[2,3,4]]
输出：-1
解释：在当前图中，Alice 无法从其他节点到达节点 4 。类似地，Bob 也不能达到节点 1 。因此，图无法完全遍历。
```
**提示：**

- `1 <= n <= 10^5`
- `1 <= edges.length <= min(10^5, 3 * n * (n-1) / 2)`
- `edges[i].length == 3`
- `1 <= edges[i][0] <= 3`
- `1 <= edges[i][1] < edges[i][2] <= n`
- 所有元组 `(typei, ui, vi)` 互不相同

#### 题目链接

<https://leetcode-cn.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/>

#### **思路:**

　　要让删除的边数最大，就要在不形成环的情况下，让公共边(类型为3的边)尽量多的保留。  

　　不考虑类型3的边，最少需要的边数量是`2×(n-1)`，然后类型3的边可以被共用。共用类型3的边时要保证不会形成环。  

　　使用并查集，插入类型为3的边，不形成环的类型为3的边数即为可共用的边数，记为`common`；  

　　考虑类型为3的公共边，最少需要的边数`need`为`2×(n-1) - common`；  

　　最多去除的边数=`总的边数`-`最少需要的边数`。

#### **代码:**

```python
# 如果不考虑-1的情况，代码如下
class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        union = UnionFind(n)
        mem = defaultdict(dict)

        for tp, u, v in edges:
            if tp == 3:
                union.union(u-1, v-1)

        common = len(list(filter(lambda x: x>0, union.uf)))
        need = 2 * (n-1) - common
        return len(edges) - need

```

其中并查集类的定义如下：  

```python
class UnionFind():
    def __init__(self, n):
        self.uf = [-1] * n  # 记录每个结点的领导，-1代表领导是自己
        self._count = n

    def find(self, x):
        r = x
        while self.uf[x] >= 0:
            x = self.uf[x]
        # 路径压缩
        while r != x:
            self.uf[r],r = x,self.uf[r]
        return x

    def union(self, x, y):
        ux, uy = self.find(x), self.find(y)
        if ux == uy:
            return
        # 规模小的优先合并
        if self.uf[ux] < self.uf[uy]:
           self.uf[ux] += self.uf[uy]
           self.uf[uy] = ux
        else:
           self.uf[uy] += self.uf[ux]
           self.uf[ux] = uy
        self._count -= 1

    def count(self):
        return self._count
```






