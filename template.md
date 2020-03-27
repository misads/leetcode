# 字符串

## A6. Z 字形变换

#### 题目描述

>　　将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。  
>　　  
>　　比如输入字符串为 "LEETCODEISHIRING" 行数为 3 时，排列如下：  

```yaml
　　L   C   I   R  
　　E T O E S I I G  
　　E   D   H   N  
```

>　　之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："LCIRETOESIIGEDHN"。  
>　　  
>　　请你实现这个将字符串进行指定行数变换的函数：  
>　　  
>　　string convert(string s, int numRows);  
>　　  
>　　示例 1:  
>　　输入: s = "LEETCODEISHIRING", numRows = 3  
>　　输出: "LCIRETOESIIGEDHN"  
>　　  
>　　示例 2:  
>　　输入: s = "LEETCODEISHIRING", numRows = 4  
>　　输出: "LDREOEIIECIHNTSG"  
>　　解释:  

```yaml
　　L     D     R  
　　E   O E   I I  
　　E C   I H   N  
　　T     S     G  
```

#### 题目链接

<https://leetcode-cn.com/problems/zigzag-conversion/>

#### **思路:**

　　找规律法，经观察发现，除了第一行和最后一行外，每一行的下一个数，要么就是从底下拐(经过2\*(numRows-1-i)个字母)，要么就是从上面拐(经过(2\*i)个字母), 用flag作为标记，是否从底下拐。

#### **代码:**

```c
/*
    执行用时 : 0 ms, 在所有 cpp 提交中击败了 100% 的用户
    内存消耗 : 10.2 MB, 在所有 cpp 提交中击败了 91.73% 的用户
*/

string convert(string s, int numRows) {
    if (numRows==1) return s;
    string ans = "";
    for(int i =0;i<numRows;i++){
        int j=i;
        bool flag=true;  // flag为true表示从底下拐，否则从上面拐
        if (i==numRows-1)flag = false;  // 第一行总是true 最后一行总是false 
        while(j<s.size()){
            ans += s[j];
            if (flag){
                j += 2*(numRows-1-i);
                if (i!=0) flag=false; // 第一行总是true
            }else{
                j+= 2 * i;
                if (i!=numRows-1) flag = true; // 最后一行总是false 
            }
        }
    }

    return ans;
}
```

