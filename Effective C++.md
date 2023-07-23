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
const修饰函数返回值代表返回值不可改变（尽可能使用const修饰不需要修改的函数返回值）。
const修饰成员函数代表该成员函数由const对象调用。

# 6.确定对象被使用前已经被初始化
永远在使用对象之前将它初始化。
* 对于内置对象手工完成初始化
* 对于自定义类，确保每个构造函数对每都将对象的每一个成员初始化
对于自定义对象的构造函数，优先使用成员初始化列表（成员初始化列表的初始化行为发生在进入当前初始化函数本体之前）。
为免除跨编译单元的初始化顺序造成的问题，请用local static对象替代non-local static对象。

# 7.了解c++默默编写并调用了哪些函数
如果你自定义一个类且没有做对应的声明且对应函数被调用，则编译器会为它声明一个default构造函数、copy构造函数、copy赋值和一个析构函数。且这些函数都是public且inline。
编译器自动生成的copy构造函数只是简单的将源对象的每一个non-static成员变量拷贝到目标对象。（会自动嵌套的调用 成员的copy构造函数。）

# 8.若不想编译器自动生成函数，应该明确的拒绝
上面说过当default的构造函数、copy构造函数和copy赋值等没有被定义但是被调用的时候，编译器会自动生成对应的函数。而在一些场景下我们不希望这几个函数被调用，自然也就不希望编译器自动生成对应函数，这个时候我们应当明确的拒绝编译器生成对应函数。
其方法有二：
1. 主动以private方式声明对应函数
```c++
//define these funcs by private
class HomeForSale {
public:
	//...
private:
	//...
	HomeForSale(const HomeForSale&);
	HomeForSale& operator=(const HomeForSale&);
}
```
2.  定义一个专门阻止copying的base class。所有需要明确拒绝编译器自动生成的类由该基类派生。
```c++
class Uncopyable{
protected:
	Uncopyable(){}
	~Uncopyable(){}
private:
	Uncopyable(const Uncopyable&);
	Uncopyable& operator=(const Uncopyable&);
}

class HomeForSale: private Uncopyable{
	//...
}
```

# 9.为多态基类声明virtual析构函数
假设如下基类和派生类
```c++
class TimeKeeper{
public:
	TimeKeeper();
	~TimeKeeper();
}

class AtomicClock:public TimeKeeper{...}

TimeKeeper* ptk = new AtomicClock();
delete ptk;
```
上述问题在于基类指针TimeKeeper* 指向了一个派生类实例，我们delete基类指针会导致未定义的行为。可能的情况是派生类可能会有自己的成员，这样的delete可能会局部销毁派生类的基类部分，形成”局部销毁“情况，最终造成资源泄露。
避免这种情况的办法是给base class一个virtual 析构函数。
```c++
```c++
class TimeKeeper{
public:
	TimeKeeper();
	virtual ~TimeKeeper(); // virtual function
}

class AtomicClock:public TimeKeeper{...}

TimeKeeper* ptk = new AtomicClock();
delete ptk;
```
除了virtual析构函数外，base class可能含有其他virtual函数。*__任何class只要带有virtual函数，都应当也有一个virtual析构函数。__* 当class不被作为base class时不应当将其析构函数设为virtual。不要继承标准容器或其他带有non virtual析构函数的class。*__纯虚析构函数需要一个空的定义。__*