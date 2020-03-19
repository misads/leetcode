# 数组

## A4. 寻找两个有序数组的中位数

#### 题目描述

>　　给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。  
>　　  
>　　请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。  
>　　  
>　　你可以假设 nums1 和 nums2 不会同时为空。  
>　　  
>　　示例 1:  
>　　nums1 = [1, 3]  
>　　nums2 = [2]  
>　　  
>　　则中位数是 2.0  
>　　  
>　　示例 2:  
>　　nums1 = [1, 2]  
>　　nums2 = [3, 4]  
>　　  
>　　则中位数是 (2 + 3)/2 = 2.5  

#### 题目链接

<https://leetcode-cn.com/problems/median-of-two-sorted-arrays/>

#### **思路**

　　将两个数组合并，然后按顺序查找即可。

#### **代码**

```python
# encoding=utf-8
"""
    执行用时 : 112 ms, 在所有 python3 提交中击败了91.33%的用户
    内存消耗 : 12.8 MB, 在所有 python3 提交中击败了99.43%的用户
"""
def findMedianSortedArrays(nums1: list, nums2: list) -> float:
    i = 0
    j = 0
    l1 = len(nums1)
    l2 = len(nums2)
    ans = []
    while i<l1 and j<l2:
        x = nums1[i]
        y = nums2[j]
        if x<=y:
            ans.append(x)
            i+=1
        else:
            ans.append(y)
            j+=1

    if i<l1:
        ans.extend(nums1[i-l1:])
    else: 
        ans.extend(nums2[j-l2:])

    # print(ans)
    l = len(ans)
    if len(ans) %2==1:
        return ans[l//2]
    else:
        return (ans[l//2-1] + ans[l//2])/2


nums2 = [1,2]
nums1 = [3,4]
print(findMedianSortedArrays(nums1,nums2))

```



## A11. 盛最多水的容器

#### 题目描述
>　　给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。  
>　　  
>　　说明：你不能倾斜容器，且 n 的值至少为 2。  
>　　  
>　　图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。  
>　　  
>　　  
>　　示例：  
>　　输入：[1,8,6,2,5,4,8,3,7]  
>　　输出：49  

#### 题目链接

https://leetcode-cn.com/problems/container-with-most-water/

#### 思路  

　　方法一：用`lefts`记录`heights`从左到右所有比之前最高的还高的线，`rights`记录从右到左所有比之前最高的还高的线。遍历`lefts`和`rights`，其中必有一种组合能够容纳最多的水。  
　　方法二：双指针，初始时设头指针和尾指针分别为`i`和`j`。我们能够发现不管是左指针向右移动一位，还是右指针向左移动一位，容器的底都是一样的，都比原来减少了 1。这种情况下我们想要让指针移动后的容器面积增大，就要使移动后的容器的高尽量大，所以我们选择指针所指的高较小的那个指针进行移动，这样我们就保留了容器较高的那条边，放弃了较小的那条边，以获得有更高的边的机会。

#### 代码  

方法一

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

方法二

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

## A15. 三数之和

#### 题目描述

>　　给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。  
>　　  
>　　注意：答案中不可以包含重复的三元组。  
>　　  
>　　  
>　　示例：  
>　　给定数组 nums = [-1, 0, 1, 2, -1, -4]，  
>　　  
>　　满足要求的三元组集合为：  
>　　[  
>　　  [-1, 0, 1],  
>　　  [-1, -1, 2]  
>　　]  

#### 题目链接

<https://leetcode-cn.com/problems/3sum/>


#### 思路  

　　记录每个数字出现的次数，如果有`多于3次`的0，或者`多于2次`的其他数，则忽略不使用。  
　　分以下几种情况分别考虑：  

　　0 + 0 + 0 = 0，0 + `一对相反数` = 0， `两个正数` + `一个负数` = 0， `两个负数` + `一个正数` = 0。

#### 代码  
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) == 0:
            return []
        times = {}  # 记录每个数出现的次数，其中0最多出现3次，其他数最多出现2次
        new_nums = []
        ans = set()
        for num in nums:
            if num not in times:
                times[num] = 1
                new_nums.append(num)
            else:
                if num == 0 and times[num] <= 2:
                    times[num] += 1
                    new_nums.append(num)
                if nums != 0 and times[num] <= 1:
                    times[num] += 1
                    new_nums.append(num)
        
        new_nums = sorted(new_nums)  # 构建一个新的数组并排序，去掉了冗余的数字

        if 0 in times:
            if times[0] == 3:  # 3个0的特例
                ans.add((0, 0, 0))
            for num in times:
                if num > 0 and -num in times:  # 0和一对相反数
                    ans.add((-num, 0, num))

        for i, num1 in enumerate(new_nums):
            for j in range(i+1, len(new_nums)):
                num2 = new_nums[j]
                if num1 < 0 and num2 < 0 and -num1-num2 in times:  # 两正一负
                    ans.add((num1, num2, -num1-num2))
                if num1 > 0 and num2 > 0 and -num1-num2 in times:  # 两负一正
                    ans.add((-num1-num2, num1, num2))

        return [i for i in ans]
```



## A16. 最接近的三数之和

#### 题目描述

>　　给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。  
>　　  
>　　例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.  
>　　  
>　　与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).  

#### 题目链接

<https://leetcode-cn.com/problems/3sum-closest/>


#### 思路  

　　维护两个集合`ones`和`twos`，前者记录不重复的数，后者通过遍历`ones`来记录任意两个数相加的和。  

　　令`delta`表示任意三个数和target之差的最小值。对于`nums`中的新的一个数字`num`，如果和`twos`中的某个数和与`target`之差小于`delta`，则更新`delta`和`ans`。`num`和`one`中所有数字的和会被加入到`twos`中。

#### 代码  
```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        delta = 9999999
        ans = -1
        ones = set()
        twos = set()
        for num in nums:
            for two in twos:  # 任意两数之和的集合
                if abs(two + num - target) < delta:
                    delta = abs(two + num - target)
                    ans = two + num
            for one in ones:  # 新数num和ones中的每个数字相加，并放入twos中
                twos.add(one+num)
            ones.add(num)  # 新数放入ones中

        return ans
```

## A26. 删除排序数组中的重复项

#### 题目描述
>　　给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。  
>　　  
>　　不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。  
>　　  
>　　  
>　　示例 1:  
>　　给定数组 nums = [1,1,2],   
>　　  
>　　函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。   
>　　  
>　　你不需要考虑数组中超出新长度后面的元素。  
>　　  
>　　示例 2:  
>　　给定 nums = [0,0,1,1,1,2,2,3,3,4],  
>　　  
>　　函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。  
>　　  
>　　你不需要考虑数组中超出新长度后面的元素。  

#### 题目链接

<https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/>


#### 思路  


　　用python自带的`remove`函数（这个解法很耗时间）。优化时间复杂度的方法可以使用双指针。

#### 代码  
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        shown = set()
        i = 0
        while i < len(nums):
            num = nums[i]
            if num in shown:
                nums.remove(num)
                i -= 1
            else:
                shown.add(num)
            i += 1

        return len(nums)
```

## A27. 移除元素

#### 题目描述
>　　给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。  
>　　  
>　　不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。  
>　　  
>　　元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。  
>　　  
>　　示例 1:  
>　　给定 nums = [3,2,2,3], val = 3,  
>　　  
>　　函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。  
>　　  
>　　你不需要考虑数组中超出新长度后面的元素。  
>　　  
>　　示例 2:  
>　　给定 nums = [0,1,2,2,3,0,4,2], val = 2,  
>　　  
>　　函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。  
>　　  
>　　注意这五个元素可为任意顺序。  
>　　  
>　　你不需要考虑数组中超出新长度后面的元素  

#### 题目链接

<https://leetcode-cn.com/problems/remove-element/>


#### 思路  


　　挑战最短代码。

#### 代码  
```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        c = nums.count(val)
        while c:
            nums.remove(val)
            c -= 1
            
        return len(nums)
```

## A31. 下一个排列

#### 题目描述
>　　实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。  
>　　  
>　　如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。  
>　　  
>　　必须原地修改，只允许使用额外常数空间。  
>　　  
>　　以下是一些例子，输入位于左侧列，其相应输出位于右侧列。  
>　　1,2,3 → 1,3,2  
>　　3,2,1 → 1,2,3  
>　　1,1,5 → 1,5,1  

#### 题目链接

<https://leetcode-cn.com/problems/next-permutation/>


#### 思路  


　　其实就是从数组倒着查找，找到`nums[i]`比`nums[i+1]`小的时候，就将`nums[i]`和`nums[i+1:]`中比`nums[i]`大的**最小的数**和`nums[i]`交换，然后再把`nums[i+1:]`排序就ok了🙆‍♂️。

#### 代码  
```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums) <= 1:
            return nums
        
        l = len(nums)
        i, j = 0, 0  # 下标为i和下标为j的数交换

        for i in range(l-2, -1, -1):
            if nums[i+1] > nums[i]:
                break
            elif i == 0:
                i = -1

        if i == -1:
            nums[:] = sorted(nums)[:]
        else:
            exchange = float('inf')
            for k, num in enumerate(nums[i+1:]):
                if num > nums[i] and num < exchange:  # 找到比nums[i]大的最小的数
                    exchange = num
                    j = k + i + 1
            nums[i], nums[j] = nums[j], nums[i]  # 下标为i和下标为j的数交换
            nums[i+1:] = sorted(nums[i+1:])
```

## A33. 搜索旋转排序数组

#### 题目描述

>　　假设按照升序排序的数组在预先未知的某个点上进行了旋转。  
>　　  
>　　( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。  
>　　  
>　　搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。  
>　　  
>　　你可以假设数组中不存在重复的元素。  
>　　  
>　　你的算法时间复杂度必须是 O(log n) 级别。  
>　　  
>　　示例 1:  
>　　输入: nums = [4,5,6,7,0,1,2], target = 0  
>　　输出: 4  
>　　  
>　　示例 2:  
>　　输入: nums = [4,5,6,7,0,1,2], target = 3  
>　　输出: -1  

#### 题目链接

<https://leetcode-cn.com/problems/search-in-rotated-sorted-array/>


#### 思路  

　　`nums`从中间切一半，必然有一半是有序的，另一半是无序的，对有序的一半二分查找，对无序的一半递归调用该算法。  
　　如果第一个数`nums[i]` 小于中间的数`nums[mid]`，则左半边有序，否则右半边有序。  

#### 代码  
```python
class Solution:
    def helper(self, nums: List[int], i, j, target):
        if j <= i:
            return -1
        n = j - i
        if n <= 2:
            for k in range(i, j):
                if nums[k] == target:
                    return k
            return -1

        middle = (i + j) // 2

        if nums[i] < nums[middle]:
            # 对左边进行二分查找，对右边递归
            start, end = middle, j
            j = middle
        else:
            # 对右边进行二分查找，对左边递归
            start, end = i, middle
            i = middle

        while i <= j and i < len(nums):
            mid = (i + j) // 2
            if nums[mid] > target:
                j = mid - 1
            elif nums[mid] < target:
                i = mid + 1
            else:
                if nums[mid] == target:
                    return mid

        return self.helper(nums, start, end, target)


    def search(self, nums: List[int], target: int) -> int:
        return self.helper(nums, 0, len(nums), target)
```

## A34. 在排序数组中查找元素的第一个和最后一个位置

#### 题目描述
>　　给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。  
>　　  
>　　你的算法时间复杂度必须是 O(log n) 级别。  
>　　  
>　　如果数组中不存在目标值，返回 [-1, -1]。  
>　　  
>　　  
>　　示例 1:  
>　　输入: nums = [5,7,7,8,8,10], target = 8  
>　　输出: [3,4]  
>　　  
>　　示例 2:  
>　　输入: nums = [5,7,7,8,8,10], target = 6  
>　　输出: [-1,-1]  

#### 题目链接

<https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/>


#### 思路  


　　用二分法查找，如果找到了一个`target`但是不是第一个`target`，继续使用二分法在它之前查找。  

#### 代码  
```python
class Solution:
    def find_first(self, nums: List[int], target: int):
        i, j = 0, len(nums)
        while i <= j and i < len(nums):
            mid = (i + j) // 2
            if nums[mid] > target:
                j = mid - 1
            elif nums[mid] < target:
                i = mid + 1
            else:
                if nums[mid] == target:
                    if mid == 0 or nums[mid-1] != target:
                        return mid
                    else:
                        j = mid - 1
                else:
                    return -1
        return -1

    def find_last(self, nums: List[int], target: int):
        i, j = 0, len(nums)
        while i <= j and i < len(nums):
            mid = (i + j) // 2
            if nums[mid] > target:
                j = mid - 1
            elif nums[mid] < target:
                i = mid + 1
            else:
                if nums[mid] == target:
                    if mid == len(nums) - 1 or nums[mid+1] != target:
                        return mid
                    else:
                        i = mid + 1
                else:
                    return -1
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        first = self.find_first(nums, target)
        last = self.find_last(nums, target)

        return [first, last]
```

## A35. 搜索插入位置

#### 题目描述
>　　给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。  
>　　  
>　　你可以假设数组中无重复元素。  
>　　  
>　　  
>　　示例 1:  
>　　输入: [1,3,5,6], 5  
>　　输出: 2  
>　　  
>　　示例 2:  
>　　输入: [1,3,5,6], 2  
>　　输出: 1  
>　　  
>　　示例 3:  
>　　输入: [1,3,5,6], 7  
>　　输出: 4  
>　　  
>　　示例 4:  
>　　输入: [1,3,5,6], 0  
>　　输出: 0  

#### 题目链接

<https://leetcode-cn.com/problems/search-insert-position/>


#### 思路  


　　二分查找，如果第`mid`个元素大于`target`，但它前一个元素小于`target`，则返回`i`。  

#### 代码  
```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        i, j = 0, len(nums)
        while i <= j and i < len(nums):
            mid = (i + j) // 2
            if nums[mid] > target:
                if mid == 0 or nums[mid-1] < target:
                    return mid 
                j = mid - 1
            elif nums[mid] < target:
                if mid == len(nums) - 1 or nums[mid+1] > target:
                    return mid + 1
                i = mid + 1
            else:
                if nums[mid] == target:
                    return mid
        
        return -1
```

## A39. 组合总合

#### 题目描述
>　　给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。  
>　　  
>　　candidates 中的数字可以无限制重复被选取。  
>　　  
>　　说明：  
>　　  
>　　所有数字（包括 target）都是正整数。  
>　　解集不能包含重复的组合。   
>　　  
>　　示例 1:  
>　　输入: candidates = [2,3,6,7], target = 7,  
>　　所求解集为:  
>　　[  
>　　  [7],  
>　　  [2,2,3]  
>　　]  
>　　  
>　　示例 2:  
>　　输入: candidates = [2,3,5], target = 8,  
>　　所求解集为:  
>　　[  
>　　  [2,2,2,2],  
>　　  [2,3,3],  
>　　  [3,5]  
>　　]  

#### 题目链接

<https://leetcode-cn.com/problems/combination-sum/>


#### 思路  


　　动态规划。`dp[i]`记录数字`i`的所有组成情况。如示例1对应`dp[2] = [[2]]`，`dp[4] = [[2, 2]]`。从`1`到`target`迭代。  

#### 代码  
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        dp = [[] for i in range(target+1)]
        for num in candidates:
            if num > target:
                continue
            dp[num] = [(num,)]  # 一个数字组成的组合
            
        for i in range(1, target+1):
            for num in candidates:
                if i-num > 0 and len(dp[i-num])>0:
                    for combine in dp[i-num]:
                        a = list(combine)
                        if num >= a[-1]:  # 确保新的组合是有序的
                            a.append(num)
                            if tuple(a) not in dp[i]:
                                dp[i].append(tuple(a))

        return dp[target]
```

## A40. 组合总和 II

#### 题目描述
>　　给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。  
>　　  
>　　candidates 中的每个数字在每个组合中只能使用一次。  
>　　  
>　　说明：  
>　　  
>　　所有数字（包括目标数）都是正整数。  
>　　解集不能包含重复的组合。   
>　　  
>　　示例 1:  
>　　输入: candidates = [10,1,2,7,6,1,5], target = 8,  
>　　所求解集为:  
>　　[  
>　　  [1, 7],  
>　　  [1, 2, 5],  
>　　  [2, 6],  
>　　  [1, 1, 6]  
>　　]  
>　　  
>　　示例 2:  
>　　输入: candidates = [2,5,2,1,2], target = 5,  
>　　所求解集为:  
>　　[  
>　　  [1,2,2],  
>　　  [5]  
>　　]  

#### 题目链接

<https://leetcode-cn.com/problems/combination-sum-ii/>


#### 思路  

　　dfs搜索，难点在于去重。  

　　方法一：用集合来去除重复出现的结果，缺点是效率较低。  

　　方法二：先排序，在每轮的`for`循环中，除了第一个元素外，不会使用和上一个重复的元素。  

#### 代码  

方法一：

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = set()
        l = len(candidates)
        visited = [0 for i in range(l)]
        def dfs(n, target):
            nonlocal l
            if n >= l or candidates[n] > target or visited[n]:
                return
            visited[n] = candidates[n]
            if candidates[n] == target:
                temp = []
                for i, vis in enumerate(visited):
                    if vis:
                        temp.append(vis)
                ans.add(tuple(sorted(temp)))

            for i in range(n+1, l):
                dfs(i, target - candidates[n])
                visited[i] = 0

        for i in range(l):
            dfs(i, target)
            visited[i] = 0

        return [i for i in ans]
```

方法二：

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()  # [1, 1, 2, 5, 6, 7, 10]
        ans = []
        l = len(candidates)

        def dfs(n, target, cur):
            nonlocal l
            for i in range(n, l):
                if i == n or candidates[i] != candidates[i-1]:  # 除了第一个元素外，不使用重复的
                    if target < candidates[i]:  # 剪枝
                        return
                    elif target == candidates[i]:
                        ans.append(cur + [candidates[i]])
                        return
                    cur.append(candidates[i])
                    dfs(i+1, target - candidates[i], cur)
                    cur.remove(candidates[i])

        dfs(0, target, [])

        return ans
```

## A41. 缺失的第一个正数

#### 题目描述
>　　给定一个未排序的整数数组，找出其中没有出现的最小的正整数。  
>　　  
>　　示例 1:  
>　　输入: [1,2,0]  
>　　输出: 3  
>　　  
>　　示例 2:  
>　　输入: [3,4,-1,1]  
>　　输出: 2  
>　　  
>　　示例 3:  
>　　输入: [7,8,9,11,12]  
>　　输出: 1  
>　　说明:  
>　　  
>　　你的算法的时间复杂度应为O(n)，并且只能使用常数级别的空间。  

#### 题目链接

<https://leetcode-cn.com/problems/first-missing-positive/>


#### 思路  

　　1、由于只能使用`O(1)`的额外空间，所以**在原数组空间上**进行操作。  
　　2、尝试从原数组构造一个`[1,2,3,4,5,6,...,n]`的数组。  
　　3、遍历数组，找到 `1<=元素<=数组长度`的元素，如`5`，将他放到应该放置的位置，即下标 4。  
　　4、遇到范围之外的数值，如`-1`或者超过数组长度的值，不交换，继续下一个。  
　　5、处理之后的数据为`[1, 2, 4, 5]`，再遍历一遍数组，`下标+1`应该是正确值，找出第一个不符合的即可。  

**疑问**：由于在`for`循环里嵌套了`while`，最差情况下的时间复杂度还是`O(n)`吗？

#### 代码  
```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            while nums[i] >= 1 and nums[i] <= len(nums) and nums[i] != nums[nums[i]-1]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        
        for i, num in enumerate(nums):
            if num != i+1:
                return i+1

        return len(nums) + 1
```

## A42. 接雨水 

#### 题目描述
>　　给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。  
>　　  
>　　上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 感谢 Marcos 贡献此图。  
>　　  
>　　示例:  
>　　输入: [0,1,0,2,1,0,1,3,2,1,2,1]  
>　　输出: 6  

#### 题目链接

<https://leetcode-cn.com/problems/trapping-rain-water/>


#### 思路  


　　先遍历一遍`height`，分别找到每个高度`h`的`左侧最高点`和`右侧最高点`，如果min(`左侧最高点`，`右侧最高点`) > h，则可以接雨水。将每个`h`接的雨水数累加。  　　

#### 代码  
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        i, j = 0, 0
        n = len(height)
        if n <= 2:
            return 0
        left_maxes = [0 for i in range(n)]  # 表示左边最高点
        right_maxes = [0 for i in range(n)]  # 表示右边最高点
        temp = height[0]
        for i in range(1, n):
            left_maxes[i] = temp
            temp = max(temp, height[i])
        temp = height[-1]
        for i in range(n-2, -1, -1):
            right_maxes[i] = temp
            temp = max(temp, height[i])

        ans = 0
        for i in range(1, n-1):  # 第一个和最后一个不可能接雨水
            h = min(left_maxes[i], right_maxes[i])
            a = max(h - height[i], 0)
            ans += a

        return ans
```

## A45. 跳跃游戏 II 

#### 题目描述
>　　给定一个非负整数数组，你最初位于数组的第一个位置。  
>　　  
>　　数组中的每个元素代表你在该位置可以跳跃的最大长度。  
>　　  
>　　你的目标是使用最少的跳跃次数到达数组的最后一个位置。  
>　　  
>　　  
>　　示例:  
>　　输入: [2,3,1,1,4]  
>　　输出: 2  
>　　解释: 跳到最后一个位置的最小跳跃数是 2。  
>　　     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。  

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

## A48. 旋转图像

#### 题目描述
>　　给定一个 n × n 的二维矩阵表示一个图像。  
>　　  
>　　将图像顺时针旋转 90 度。  
>　　  
>　　说明：  
>　　  
>　　你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。  
>　　  
>　　  
>　　示例 1:  
>　　给定 matrix =   
>　　[  
>　　  [1,2,3],  
>　　  [4,5,6],  
>　　  [7,8,9]  
>　　],  
>　　  
>　　原地旋转输入矩阵，使其变为:  
>　　[  
>　　  [7,4,1],  
>　　  [8,5,2],  
>　　  [9,6,3]  
>　　]  
>　　  
>　　示例 2:  
>　　给定 matrix =  
>　　[  
>　　  [ 5, 1, 9,11],  
>　　  [ 2, 4, 8,10],  
>　　  [13, 3, 6, 7],  
>　　  [15,14,12,16]  
>　　],   
>　　  
>　　原地旋转输入矩阵，使其变为:  
>　　[  
>　　  [15,13, 2, 5],  
>　　  [14, 3, 4, 1],  
>　　  [12, 6, 8, 9],  
>　　  [16, 7,10,11]  
>　　]  

#### 题目链接

 <https://leetcode-cn.com/problems/rotate-image/>


#### 思路  

　　<img src='_img/48.png' style="zoom:50%;">  
　　扣四个边界出来。四个边界对应的点交换。每遍历一层，就往里缩一个矩阵。  

#### 代码  
```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = matrix
        n = len(m) - 1
        for l in range((n+1) // 2):  # 从外往里第几层
            for i in range(n - l * 2):
                m[l][l+i], m[i+l][n-l], m[n-l][n-l-i], m[n-l-i][l] =  m[n-l-i][l], m[l][l+i], m[l+i][n-l], m[n-l][n-l-i] 

```