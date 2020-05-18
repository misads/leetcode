# 杂项

## A146. LRU缓存机制

难度`中等`

#### 题目描述

运用你所掌握的数据结构，设计和实现一个  [LRU (最近最少使用) 缓存机制](https://baike.baidu.com/item/LRU)。它应该支持以下操作： 获取数据 `get` 和 写入数据 `put` 。

获取数据 `get(key)` - 如果密钥 (key) 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
写入数据 `put(key, value)` - 如果密钥已经存在，则变更其数据值；如果密钥不存在，则插入该组「密钥/数据值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
**进阶:**

你是否可以在 **O(1)** 时间复杂度内完成这两种操作？
> **示例:**

```
LRUCache cache = new LRUCache( 2 /* 缓存容量 */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // 返回  1
cache.put(3, 3);    // 该操作会使得密钥 2 作废
cache.get(2);       // 返回 -1 (未找到)
cache.put(4, 4);    // 该操作会使得密钥 1 作废
cache.get(1);       // 返回 -1 (未找到)
cache.get(3);       // 返回  3
cache.get(4);       // 返回  4
```

#### 题目链接

<https://leetcode-cn.com/problems/lru-cache/>

#### **思路:**

　　区别`LRU`和`LFU`：  


　　`LRU`是**最近最少使用页面**置换算法(`Least Recently Used`),也就是首先淘汰**最长时间未被使用**的页面！  

　　`LFU`是**最近最不常用页面**置换算法(`Least Frequently Used`),也就是淘汰**一定时期内被访问次数最少的页**!  

　　`LRU`关键是看页面**最后一次被使用**到**发生调度**的时间长短；  

　　而`LFU`关键是看**一定时间段内页面被使用的频率**!　　

#### **代码:**

```python
class LRUCache:
    def __init__(self, capacity: int):
        self.mem = {}
        self.times = {}
        self.time = 0
        self.capacity = capacity

    def get(self, key: int) -> int:
        self.time += 1
        if key in self.mem:
            self.times[key] = self.time
            return self.mem[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        self.time += 1
        self.times[key] = self.time
        self.mem[key] = value

        if len(self.mem) > self.capacity:
            minimal = min(self.times, key=self.times.get)
            self.mem.pop(minimal)
            self.times.pop(minimal)
        # print(self.times)

```
