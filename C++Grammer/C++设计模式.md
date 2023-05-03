# 1.单例模式
单例模式(Singleton Pattern，也称为单件模式)，使用最广泛的设计模式之一。其意图是保证一个类仅有一个实例，并提供一个访问它的全局访问点，该实例被所有程序模块共享。
定义一个单例类：
+ 私有化它的构造函数，以防止外界创建单例类的对象；
+ 使用类的私有静态指针变量指向类的唯一实例；
+ 使用一个公有的静态方法获取该实例。
下面是一个懒汉版的单例模式
```
#include <iostream>
 
using namespace std;
 
class Singleton{
private:
    static Singleton* m_pInstance;
private:
    Singleton(){
        cout << "constructor called!" << endl;
    }
    Singleton(Singleton&)=delete;
    Singleton& operator=(const Singleton&)=delete;
public:
    ~Singleton(){
        cout << "destructor called!" << endl;
    }
    static Singleton* getInstance() {
        if (m_pInstance == nullptr) {
            m_pInstance = new Singleton;
        }
        return m_pInstance;
    }
};
 
Singleton* Singleton::m_pInstance = nullptr;
 
int main() {
    Singleton* instance1 = Singleton::getInstance();
    Singleton* instance2 = Singleton::getInstance();
 
    return 0;
}
```
上述懒汉式的单例模式将类的构造函数私有化，通过一个静态的共有函数getInstance()来进行类的一次实例化，当类本身以及存在的时候则直接返回类本身。

上述代码中我们发现类的私有数据中有一个用类本身定义的指针，这是可行的。（类本身定义的指针是可行的，类本身定义的类则是不可行的。因为编辑器在定义一个类的时候需要知道这个类占用内存大小，而无论这个类本身多大，其定义的指针类型占用的内存大小是固定的。）

懒汉式的单例模式存在两个问题
1. 内存泄漏问题（只有new没有delete）
2. 多线程场景下的线程安全问题（考虑加锁）

最推荐的懒汉式单例是通过局部静态变量来实现，由《Effective C++》作者提出。
```
class Singleton
{
public:
    ~Singleton(){
        std::cout<<"destructor called!"<<std::endl;
    }
 
    Singleton(const Singleton&)=delete;
    Singleton& operator=(const Singleton&)=delete;
 
    static Singleton& getInstance(){
        static Singleton instance;
        return instance;
    }
 
private:
    Singleton(){
        std::cout<<"constructor called!"<<std::endl;
    }
};
 
int main(int argc, char *argv[])
{
    Singleton& instance_1 = Singleton::getInstance();
    Singleton& instance_2 = Singleton::getInstance();
 
    return 0;
}
```
这个版本的单例模式利用的是局部静态变量初始化的标准特性来保证线程安全，静态变量的生存周期保证了是一种单例模式。
上述也可以返回指针而不是引用。
```
static Singleton* get_instance(){ 
	static Singleton instance; 
	return &instance; 
}
```