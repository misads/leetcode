# 一些tricks

## 递归 & 搜索

- 可以用`sys.setrecursionlimit(100000)`自定义最大递归深度（默认为`998`）。
- dfs时，如果有重复的元素，可以先排序后用下面的方法剪枝：

```python
for i in range(n, l):
    if i == n or nums[i] != nums[i-1]:  # 用不同递归次数来减枝
        temp.append(nums[i])
        dfs(i+1, temp)
        temp.remove(nums[i])
      
```

- dfs搜索的位置**可能出现重复**时，需要使用`visited`数组。

## 动态规划


- 动态规划的时候将空的情况也放在dp数组中考虑，有时可以简化问题。
- 通配符`'*'`匹配字符串时，转移方程`dp[j][i]`=`dp[j][i-1]`or`dp[j-1][i]`。
- 动态规划一般只考虑`dp[i]`和`dp[i-1]`之间的关系，尽量不要用max(`dp[:i]`)这样的关系。
- 动态规划的缓存写法：

```python
from functools import lru_cache
@lru_cache(None)
def dp(i,j):
    pass
```
- 动态规划中的**切割**问题。只有某半边符合要求时才有判断另外半边的必要。  

## 字符串

- 双指针的写法：

```python
left = 0
for right, char in enumerate(s):
    pass
```

- 统计字符个数

```python
import collections
# a = 'aaabbbcc'
c = collections.Counter(a)
# c = Counter({'a': 3, 'b': 3, 'c': 2})
```

- 判断两个字符串字符是否相同

```python
# a = "abca"  b = "caab"
sorted(a) == sorted(b)
```

## 数组


- 创建矩阵

```python
# 创建m×n的矩阵
m = [[0 for i in range(n)] for j in range(m)]
```

- 两数组对应元素相加：

```python
# a = [1, 2, 3] b = [4, 5, 6] c = [5, 7, 9]
c = [sum(i) for i in zip(a, b)]
```
- 两数组合并：

```python
def merge(tmp1, tmp2):
    k = len(tmp1) + len(tmp2)
    return [max(tmp1, tmp2).pop(0) for _ in range(k)]
  
# merge([5, 3, 1], [2, 4, 6]) = [5,3,2,4,6,1]
```

- 二分查找插入的位置

```python
import bisect
idx = bisect.bisect(asc, num)  # 查找应该插入的位置
idx = bisect.bisect_left(asc, num)  #  有重复元素时找到最左边的位置
bisect.insort(asc, num)  # 插入
bisect.insort_left(asc, num)

# 从升序数组asc中删除num(复杂度O(logn))
idx = bisect.bisect_left(asc, num); asc.pop(idx)

```

## 字典

- 有默认值的字典

```python
from collections import defaultdict
mem = defaultdict(int)
```

## 排序

- 自定义比较函数

```python
from functools import cmp_to_key

def cmpp(s1, s2):  # 定义一个比较函数
    if int(s1 + s2) > int(s2 + s1):  # 也可以 return int(s1 + s2) - int(s2 + s1)
        return 1
    elif int(s1 + s2) < int(s2 + s1):
        return -1
    else:
        return 0

nums.sort(key=cmp_to_key(cmpp))

```

## 数学

- 最大公约数 `math.gcd(a, b)`

## 链表&树

- 链表逆序

```python
node = head
rever = None
while node:
    node.next, rever, node = rever, node, node.next
```

## 位运算

- `i & (i - 1)`可以去掉i最右边的一个1（如果有）。
- `i >> 1`会把最低位去掉。  
- `x & (-x)`，称作lowbit，当x为0时结果为0；x为奇数时，结果为1；x为偶数时，结果为x中2的最大次方的因子。

## 其他

- 处理环的时候可以从某一点裂开，然后分几种情况分别讨论。
- **原地算法**。一般的处理方法是用原空间的一些位置来做标记。
