# 1. 优先使用explicit声明构造函数
explicit声明的构造函数会阻止编译器进行隐式类型转换，在参数传递时隐式的类型转换可能带来不容易发现的风险。所以用explicit声明自定义类的构造函数是个好习惯，除非我们有一个好理由允许构造函数进行隐式类型转换。

# 2.区分copy构造函数和copy赋值
copy构造函数被用来“以同类型对象初始化自我对象”，而copy赋值被用来“从另一个同类型对象中拷贝其值到自我对象”。
可以看出，当有一个新的对象产生的时候就一定会有copy构造函数被调用。函数实参到形参的传值过程中，passed by value方式会调用**copy构造函数**。如下述代码所示：
```c++
bool hasAcceptableQuality(Widget w);
...
Widget aWidget;
if(hasAcceptableQuality(aWidget))...
```
* 对于C内置类型而言采用pass by value通常比pass by reference更高效
* 对于用户自定义类型采用pass by reference to const更好

# 3.把C++看作一个语言联邦：C语言、Object-Oriented C++、Template C++、STL

# 4.尽量用const,enum,inline替换#define
宏定义在预处理阶段就会被替换，并不会被编译器真正“看到”，所以会引入很多潜在风险。因此更好的办法是采用const常量替代宏定义的常量。
在用常量替换#define时有两个情况需注意：1.常量指针不能保证指向的内容不变，除非将const写两遍```const char* const authorName = "Scott Meyers"``` （指针及指向内容都不变。）2.class专属常量需要声明为static以确保只实例化一份实体。
也可以使用enum类型当作一个整型常量。如下：
```c++
class GamePlayer{
private:
	enum{NumTurns = 5};
	int scoresp[NumTurns];
};
```
函数式的宏可以用inline来替换：
```c++
#define CALL_WITH_MAX(a, b) f((a) > (b)? (a):(b))
// ||  ||  ||
// \/  \/  \/
template<typename T>
inline void callWithMax(const T&a, const T&b){
	f(a>b? a: b);
}
``` 
# 5.尽可能使用const
const修饰的变量其值不可以被改变。
当const修饰指针的时候，const出现在星号左边则代表指针指向内容不可改变，当const出现在星号右边则代表指针变量本身不可改变。
const修饰函数返回值代表返回值不可改变。
const修饰成员函数代表该成员函数由const对象调用
