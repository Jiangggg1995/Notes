# C++ Grammar

## 宏

### 什么是宏？

它是一种预处理指令，在预编译阶段将宏名替换成后面的替换体。

```c++
#define <宏名> <宏体>
#define <宏名>(<参数表>) <宏体>
宏定义可以使用多行,需要添加 \ 符号。
```

### 为什么使用宏？

1. 可以提高代码可读性和可维护性。
2. 避免函数调用，提高程序执行效率。

### 用法示例

#### 1.简单宏

```c++
#include "iostream"
#define N 2+2
int main(){
    using namespace std;
    int a = N*N;
    cout << a << endl;
    return 0;
}
```

输出: *8*

解释：a = 2+2*2+2

#### 2.带参宏

```c++
#include "iostream"
#define area(x) x*x
int main(){
    using namespace std;
    int y = area(2+2);
    cout << y << endl;
    return 0;
}
```

输出： *8*

解释：同上

#### 3.可变宏

```c++
#include "stdio.h"
#define DEBUG(...) printf(__VA_ARGS__)
#define ERROR(fmt,...) printf(fmt,__VA_ARGS__)
int main(){
    int y = 555;
    DEBUG("Y=%d\n", y);
    ERROR("a = %d,b = %d,c = %d\n", 1, 2, 3);
    return 0;
}
```

输出:   *Y=555*
             *a = 1,b = 2,c = 3*

解释：`__VA_ARGS__` 代替参数列表里的`...`

#### 4.特殊宏

C/C++里面有一些特殊宏：\_\_FUNCTION\_\_, \_\_FILE\_\_, \_\_LINE\_\_.其用法如下：

```cpp
// this file is in /mnt/d/Code/learn_c/main.cpp
#include <iostream>

void hello(){
    std::cout << "__FILE__ is the file name:" << __FILE__ << std::endl;
    std::cout << "__FUNCTION__ is the function name:" << __FUNCTION__ << std::endl;
    std::cout << "__LINE__ is the line: " << __LINE__ << std::endl;
}

int main()
{
    using namespace std;
    hello();
    return 0;
}
```

输出：\_\_FILE\_\_ is the file name:/mnt/d/Code/learn_c/main.cpp
            \_\_FUNCTION\_\_ is the function name:hello
            \_\_LINE\_\_ is the line: 6

## 类成员冒号初始化以及构造函数内赋值

todo  

 [C++类成员冒号初始化以及构造函数内赋值_zj510的专栏-CSDN博客_构造函数冒号初始化](https://blog.csdn.net/zj510/article/details/8135556)

## lambda expression

### 什么是lambda expression

lambda表达式是C++11中引入的，它是一种匿名函数，通常它作为一个参数传递给接收函数指针或者函数符的函数使用。

```c++
[capture list] (params list) mutable exception-> return type { function body }

//[capture list]:捕获外部变量列表
//(params list):形参列表
//mutable:表示能不能修改捕获的变量
//exception:异常设定
//return type:返回类型
//function body:函数体
//虽然lambda中的参数变量很多，通常情况下并不需要把每一个都使用上
//大多数情况下可以直接省略->、mutable、exception，不需要返回类型的话也可以省略return type。
//一个简单的lambda表达式：[](int x){return x*x;}。
```

### 为什么使用lambda expression

1. 就近原则：随时定义随时使用，lambda表达式的定义和使用在同一个地方，并且lambda表达式可以直接在其他函数中定义使用，其他函数没有这个优势。

2. 简洁明了：lambda表达式相比较其他函数的更加的简洁明了。

3. 效率相对高些：lambda表达式不会阻止编译器的内联，而函数指针则会阻止编译器内联。

4. 捕获动态变量：lambda表达式可以捕获它可以访问的作用域内的任何动态变量。

### 用法示例

```c++
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void myPrintf(int elem){
    cout << elem << " ";
}

int main(){
    vector<int> tmpVector(10);
    int num = 1;
    generate(tmpVector.begin(), tmpVector.end(), [&num]() {
        num += num;
        return num;
    });
    cout << "--打印tmpVector中的值--" << endl;
    for_each(tmpVector.begin(), tmpVector.end(), myPrintf);
    cout << endl;

    cout << "--给lambda表达指定一个名字--" << endl;
    int a = 100;
    //bFun是为这个lambda表达式起名字
    auto bFun = [a]()->int {return a / 10; };
    int c = bFun();
    cout << "c=" << c << endl;

    cout << "--lambda表达传形参--" << endl;
    int countIndex = count_if(tmpVector.begin(), tmpVector.end(),[](int x) {return x / 200 == 0;});
    cout << "统计小于200的数的个数：" << countIndex << endl;

    cout << "--lambda表达默认捕获外部变量--" << endl;
    int tmpNum1 = 10;
    int tmpNum2 = 5;
    for_each(tmpVector.begin(), tmpVector.end(), [=](int x){
        x = x * tmpNum1 + tmpNum2;
        cout << "x=" << x << " ";
    });
    cout << endl;
    for_each(tmpVector.begin(), tmpVector.end(), myPrintf);
    cout << endl;
}
```

输出：

*-----------打印tmpVector中的值----------*
*2 4 8 16 32 64 128 256 512 1024* 
*-----------给lambda表达指定一个名字----------*
*c=10*
*-----------lambda表达传形参----------*
*统计小于200的数的个数：7*
*-----------lambda表达默认捕获外部变量,形参以值传递方式----------*
*x=25 x=45 x=85 x=165 x=325 x=645 x=1285 x=2565 x=5125 x=10245* 
2 4 8 16 32 64 128 256 512 1024* 

注意：`[=]`可以是`=`或`&`，表示`{}`中用到的、定义在`{}`外面的变量在`{}`中是否允许被改变。`=`表示不允许，`&`表示允许。

---

## 类型转换
