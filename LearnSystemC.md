# 前言

SystemC是基于C++的编程语言。SystemC在C++的基础上增加了一些重要概念，如并发、定时事件和数据类型等概念。SystemC还增加了一个类库，扩展了C++的能力，该类库提供了功能强大的新机制，这种机制可以为具有硬件时序、并发和响应行为的系统结构建模。

SystemC既是系统级语言，也是硬件描述语言，通过SystemC就能为硬件和软件系统建模。SystemC是硬件描述语言，这种语言可以为RTL级的设计建模；SystemC也可以为自己设计的整个系统建模，就像编写软件程序那样，描述该系统的行为。因为SystemC是一种既可以定义硬件组件，又可以定义软件组件的语言，使用SystemC可以无缝地进行软件和硬件的协同仿真。

虽然SystemC可以用于描述硬件，但相比Verilog或VHDL并无优势，仿真速度可能一致或相差无几。SystemC的优势在于可以在更高抽象级别上描述设计，例如用SystemC将设计描述成与时间无关或松散时间的模型，这样就不用像RTL级模型一样把每一个周期下的状态都表现出来，这样就可以显著地加快仿真速度。

本篇SystemC简单教程翻译自[Learn SystemC](https://learnsystemc.com/)，如有疏漏，欢迎指正。

作者： 江运衡
邮箱： yunheng_jiang@outlook.com

# Hello World

在将字符串打印到控制台时，有两种方法：

1. c++风格：从普通的c++函数中打印
2. systemc风格：从systemC仿真内核调用systemC方法中打印

在我们开始之前，您需要了解一些基础知识。

SystemC头文件：
要使用SystemC类库特性，一个应用程序应该包含以下C++头文件之一：

1. #inlcude<systemc.h>
    a. systemc.h将sc_core和sc_dt命名空间内的所有声明包含，以及sc_unnamed和标准C或C++库（例如cin、cout、iostream）中的选定名称。
    b. systemc.h是为了向后兼容早期的SystemC版本，并可能在将来版本的此标准中被废弃。
2. #include \<systemc\>
    a. systemc是去掉usings的旧systemc.h

```c++
#include <systemc>          //include the systemC header file
using namespace sc_core;    //use namespace
void hello1() {             //a normal c++ function
  std::cout << "Hello world using approach 1" << std::endl;
}
struct HelloWorld : sc_module {          //define a systemC module
  SC_CTOR(HelloWorld) {                  //constructor function,to be explained later
    SC_METHOD(hello2);                   //register a member function to the kernel
  }
  void hello2(void) {                    //a function for systemC simulation kernel
    std::cout << "Hello world using approach 2" << std::endl;
  }
};
int sc_main(int, char*[]) {
  hello1();
  HelloWorld helloworld("helloworld");   //instantiate a systemC module
  sc_start();                            //let systemC simulation kernel to invoke helloworld.hello2()
  return 0;
}
```

上面代码输出结果:
***Hello world using approach 1***
***Hello world using approach 2***

# SystemC Module

一个SystemC Module是：

1. 具有状态、行为和层次结构的功能的最小container。
2. 一个C++类，继承自systemC的basic class：___sc_module___
3. SystemC的主要结构构建块。
4. 用于表示实际系统中的组件。

如何定义一个系统C模块：

1. _SC_MODULE(module_name) {}_：使用SystemC定义的宏“SC_MODULE”。
2. _struct module_name: public sc_module {}_：继承自sc_module的结构体。
3. _class module_name : public sc_module { public: }_：继承自sc_module的类。
   注意，类与结构体的默认访问控制模式不同，struct默认访问权限为public，class默认访问权限为private。

如何使用一个SystemC Module：

1. sc_module类的对象只能在设计期间构造，在仿真期间实例化模块是错误的。

2. 从sc_module直接或间接派生的每个类都应至少有一个构造函数。每个构造函数应有一个且仅有一个sc_module_name类的参数，但可以具有除sc_module_name之外的其他类的更多参数。

3. 应向每个模块实例的构造函数传递一个字符串值参数。如果存在这样的变量，则最好将此字符串名称设置为与C++变量名称相同的名称，通过该变量引用模块。

4. （稍后解释）通常应调用interface方法实现模块之间的通信；即模块应通过其ports与环境通信。其他通信机制也是允许的，比如用于调试。

```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(MODULE_A) { // approach #1, use systemC provided SC_MODULE macro
  SC_CTOR(MODULE_A) { // default constructor
    std::cout << name() << " constructor" << std::endl; // name() returns the object name, which is provided during instantization
  }
};
struct MODULE_B : public sc_module { // approach #2, this uses c++ syntax and is more readiable
  SC_CTOR(MODULE_B) {
    std::cout << name() << " constructor" << std::endl;
  }
};
class MODULE_C : public sc_module { // approach #3, use class instead of struct
public: // have to explicitly declare constructor function as public 
  SC_CTOR(MODULE_C) {
    std::cout << name() << " constructor" << std::endl;
  }
};

int sc_main(int, char*[]) { // systemC entry point
  MODULE_A module_a("module_a"); // declare and instantiate module_a, it's common practice to assign module name == object name
  MODULE_B module_b("modb"); // declare and instantiate module_b, module name != object name
  MODULE_C module_c("module_c"); // declare and instantiate module_c
  sc_start(); // this can be skipped in this example because module instantiation happens during elaboration phase which is before sc_start
  return 0;
}
```

上面代码输出结果:
***module_a constructor***
***modb constructor***
***module_c constructor***

# Constructor: SC_CTOR

每个C++类必须有一个构造函数。对于一个普通的C++类，如果没有明确提供构造函数，则会自动生成默认构造函数。
然而，每个SystemC模块必须有一个唯一的“name”，该名称在实例化模块对象时提供。这需要构造函数至少有一个参数。

SystemC提供了一个方便的宏（SC_CTOR），用于声明或定义模块的构造函数。
SC_CTOR：

1. 仅在C++规则允许声明构造函数的地方使用，可以当作构造函数的声明或者定义。
2. 只有一个参数，即正在构造的模块类的name。
3. 不能向构造函数添加用户定义的参数。如果应用程序需要传递额外的参数，则必须显示提供构造函数而不是使用这个宏。

```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(MODULE_A) {
  SC_CTOR(MODULE_A) { // constructor taking only module name
    SC_METHOD(func_a); // register member function to systemC simulation kernel, to be explained later.
  }
  void func_a() { // a member function with no input, no output
    std::cout << name() << std::endl;
  }
};

SC_MODULE(MODULE_B) {
  SC_CTOR(MODULE_B) { // constructor
    SC_METHOD(func_b); // register member function
  }
  void func_b(); // declare function
};
void MODULE_B::func_b() { // define function outside class definition
  std::cout << this->name() << std::endl;
}
SC_MODULE(MODULE_C) { // constructor taking more arguments
  const int i;
  SC_CTOR(MODULE_C); // SC_HAS_PROCESS is recommended, see next example for details
  MODULE_C(sc_module_name name, int i) : sc_module(name), i(i) { // explcit constructor
    SC_METHOD(func_c);
  }
  void func_c() {
    std::cout << name() << ", i = " << i << std::endl;
  }
};

int sc_main(int, char*[]) {
  MODULE_A module_a("module_a");
  MODULE_B module_b("module_b");
  MODULE_C module_c("module_c",1);
  sc_start();
  return 0;
}
```

上面代码输出结果:
***module_a***
***module_b***
***module_c, i=1***

# SC_HAS_PROCESS

SC_HAS_PROCESS是在systemC v2.0中引入的。它只有一个参数，即模块类的名称。它通常与SC_CTOR进行比较。让我们看一下这两个宏是如何定义的：

1. SC_SCOR：

```c++
   #define SC_CTOR(user_module_name)                                             \
       typedef user_module_name SC_CURRENT_USER_MODULE;                          \
       user_module_name( ::sc_core::sc_module_name )
```

2. SC_HAS_PROCESS：

```c++
   #define SC_HAS_PROCESS(user_module_name)                                      \
       typedef user_module_name SC_CURRENT_USER_MODULE
```

当向SC_CTOR和SC_HAS_PROCESS提供“module”作为输入参数时，它们展开为：

1. SC_CTOR(module)：
   
   ```c++
   typedef module SC_CURRENT_USER_MODULE; 
   module( ::sc_core::sc_module_name )
   ```

2. SC_HAS_PROCESS(module)：
   
   ```c++
   typedef module SC_CURRENT_USER_MODULE;
   ```

从这里你可以看到：

1. 这两个宏都将“module”定义为“SC_CURRENT_USER_MODULE”，这是在通过SC_METHOD/SC_THREAD/SC_CTHREAD向simulation kernel注册成员函数时使用的。
2. SC_CTOR还声明了一个默认构造函数，该函数只有一个模块名作为输入参数。其影响是：  
    a) SC_CTOR节省了一行代码来编写构造函数文件，而如果使用SC_HAS_PROCESS，则必须声明构造函数函数：  `module_class_name(sc_module_name name, additional argument ...);  
    b) 由于SC_CTOR具有构造函数函数声明，因此它只能放在类声明的头部。

我的建议是：

1. 如果一个模块没有仿真进程，那么久不要使用SC_CTOR或SC_HAS_PROCESS（成员函数是通过SC_METHOD/SC_THREAD/SC_CTHREAD向simulation kernel注册的）。

2. 如果模块不需要其他参数（除了模块名）进行实例化，请使用SC_CTOR。

3. 当在实例化过程中需要其他参数时，请使用SC_HAS_PROCESS。

```c++
#include <systemc>
using namespace sc_core;
// module without simulation processes doesn't need SC_CTOR or SC_HAS_PROCESS
SC_MODULE(MODULE_A) { 
  // c++ style constructor, the base class is implicitly instantiated with module name.
  MODULE_A(sc_module_name name) {
  std::cout << this->name() << ", no SC_CTOR or SC_HAS_PROCESS" << std::endl;
  }
};
SC_MODULE(MODULE_B1) { // constructor with module name as the only input argument
  SC_CTOR(MODULE_B1) { // implicitly declares a constructor of MODULE_B1(sc_module_name)
    SC_METHOD(func_b); // register member function to simulation kernel
  }
  void func_b() { // define function
  std::cout << name() << ", SC_CTOR" << std::endl;
  }
};

SC_MODULE(MODULE_B2) { // constructor with module name as the only input argument
  SC_HAS_PROCESS(MODULE_B2); // no implicit constructor declarition
  // explicit constructor declaration, 
  //also instantiate base class by default via sc_module(name)
  MODULE_B2(sc_module_name name) { 
    SC_METHOD(func_b); // register member functi
  }
  void func_b() { // define function
    std::cout << name() << ", SC_HAS_PROCESS" << std::endl;
  }
};
SC_MODULE(MODULE_C) { // pass additional input argument(s)
  const int i;
  // OK to use SC_CTOR, which will also define an un-used constructor:
  // MODULE_A(sc_module_name);
  SC_HAS_PROCESS(MODULE_C);
  MODULE_C(sc_module_name name, int i) : i(i) { // define the constructor function
    SC_METHOD(func_c); // register member function
  }
  void func_c() { // define function
    std::cout << name() << ", additional input argument" << std::endl;
  }
};

SC_MODULE(MODULE_D1) { // SC_CTOR inside header, constructor defined outside header
  SC_CTOR(MODULE_D1);
  void func_d() {
    std::cout << this->name() << ", SC_CTOR inside header, constructor defined outside header" << std::endl;
  }
};
// defines constructor. Fine with/without "sc_module(name)"
MODULE_D1::MODULE_D1(sc_module_name name) : sc_module(name) { 
  SC_METHOD(func_d);
}

SC_MODULE(MODULE_D2) { // SC_HAS_PROCESS inside header, constructor defined outside header
  SC_HAS_PROCESS(MODULE_D2);
  MODULE_D2(sc_module_name); // declares constructor
  void func_d() {
    std::cout << this->name() << ", SC_CTOR inside header, constructor defined outside header" << std::endl;
  }
};
MODULE_D2::MODULE_D2(sc_module_name name) : sc_module(name) { // defines constructor. Fine with/without "sc_module(name)"
  SC_METHOD(func_d);
}

SC_MODULE(MODULE_E) { // SC_CURRENT_USER_MODULE and constructor defined outside header
  MODULE_E(sc_module_name name); // c++ style constructor declaration
  void func_e() {
    std::cout << this->name() << ", SC_HAS_PROCESS outside header, CANNOT use SC_CTOR"       << std::endl;
  }
};
MODULE_E::MODULE_E(sc_module_name name) { // constructor definition
  SC_HAS_PROCESS(MODULE_E); // NOT OK to use SC_CTOR
  SC_METHOD(func_e);
}

int sc_main(int, char*[]) {
  MODULE_A module_a("module_a");
  MODULE_B1 module_b1("module_b1");
  MODULE_B2 module_b2("module_b2");
  MODULE_C module_c("module_c", 1);
  MODULE_D1 module_d1("module_d1");
  MODULE_D2 module_d2("module_d2");
  MODULE_E module_e("module_e");
  sc_start();
  return 0;
}
```

上面代码输出结果:
***module_a, no SC_CTOR or SC_HAS_PROCESS***
***module_b1, SC_CTOR***
***module_b2, SC_HAS_PROCESS***
***module_c, additional input argument***
***module_d1, SC_CTOR inside header, constructor defined outside header***
***module_d2, SC_CTOR inside header, constructor defined outside header***
***module_e, SC_HAS_PROCESS outside header, CANNOT use SC_CTOR***

# Simulation Process

一个仿真进程：

1. 是 sc_module 类的一个成员函数
2. 没有输入参数并且不返回值
3. 已注册到simulation kernel

如何注册一个模拟过程：

1. SC_METHOD(func)：没有自己的执行线程，不消耗仿真时间，不能被挂起，并且不能调用wait() 
2. SC_THREAD(func)：有自己的执行线程，可能消耗仿真时间，可以被挂起，并且可以调用 wait() 
3. SC_CTHREAD(func, event)：SC_THREAD 的一个特殊形式，只能有一个静态的时钟边沿事件的敏感性

何时可以注册：

1. 在构造函数内
2. 在模块的 before_end_of_elaboration 或 end_of_elaboration 回调中
3. 在从构造函数或成员函数的回调中

限制：

1. 只能在同一模块的成员函数上进行注册。
2. SC_CTHREAD 不得从 end_of_elaboration 回调中调用。

注意：

1. SC_THREAD 可以执行 SC_METHOD 或 SC_CHTEAD 执行的所有操作。在示例中，我大多会使用此。
2. 为了使 SC_THREAD 或 SC_CTHREAD 进程再次被调用，必须有一个 while 循环来确保它永远不会退出。
3. SC_THREAD 进程不需要 while 循环。它通过 next_trigger() 被再次调用。
4. systemC 中的仿真时间不是程序运行的实际时间。它是由仿真内核管理的计时，稍后解释。

```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(PROCESS) {
  sc_clock clk; // declares a clock
  SC_CTOR(PROCESS) : clk("clk", 1, SC_SEC) { // instantiate a clock with 1sec periodicity
    SC_METHOD(method); // register a method
    SC_THREAD(thread); // register a thread
    SC_CTHREAD(cthread, clk); // register a clocked thread
  }
  void method(void) { // define the method member function
    // no while loop here
    std::cout << "method triggered @ " << sc_time_stamp() << std::endl;
    next_trigger(sc_time(1, SC_SEC)); // trigger after 1 sec
  }
  void thread() { // define the thread member function
    while (true) { // infinite loop make sure it never exits 
      std::cout << "thread triggered @ " << sc_time_stamp() << std::endl;
      wait(1, SC_SEC); // wait 1 sec before execute again
    }
  }
  void cthread() { // define the cthread member function
    while (true) { // infinite loop
      std::cout << "cthread triggered @ " << sc_time_stamp() << std::endl;
      wait(); // wait for next clk event, which comes after 1 sec
    }
  }
};

int sc_main(int, char*[]) {
  PROCESS process("process"); // init module
  std::cout << "execution phase begins @ " << sc_time_stamp() << std::endl;
  sc_start(2, SC_SEC); // run simulation for 2 second
  std::cout << "execution phase ends @ " << sc_time_stamp() << std::endl;
  return 0;
}
```

上面代码输出结果:
***execution phase begins @ 0 s
method triggered @ 0 s
thread triggered @ 0 s
cthread triggered @ 0 s
method triggered @ 1 s
thread triggered @ 1 s
cthread triggered @ 1 s
execution phase ends @ 2 s***
# Simulation Stages
SystemC应用程序的运行分为三个阶段：

1. 推导（Elaboration）：在sc_start()之前执行语句。 
    主要目的是创建支持仿真语义的内部数据结构。 在推导过程中，模块层次结构的部分（模块、端口、原始通道和进程）被创建，ports和exports绑定到通道上。
2. 执行（Execution）：进一步细分为两个阶段： 
    a) 初始化
        simulation kernel识别所有模拟进程，并将它们置于可运行或等待的进程集合之中。 除请求“无初始化”的进程外，所有仿真进程都位于可运行集合中。 
    b) 仿真 
        通常描述为一个状态机，它调度进程运行并推进仿真时间。它有两个内部阶段：
            1) 评估：一次运行所有可运行进程。每个进程运行直到遇到wait()语句或return语句。如果没有可运行进程，则停止。
            2) 推进时间：一旦可运行进程集为空，仿真将进入推进时间阶段，此时它会执行以下操作：
                a) 将仿真时间移动到最接近预定事件的时间 
                b) 将一些等待该特定时间的进程放到可运行集合中 
                c) 返回到评估阶段 
            从评估到推进时间的进展一直持续到发生以下三种情况之一。然后它将进入清理阶段。 
                a) 所有进程都已让出 
                b) 一个进程已执行sc_stop() 
                c) 达到最大时间

3. 清理或后处理：销毁对象、释放内存、关闭打开的文件等。

在推导和仿真过程中，内核在不同的阶段调用四个回调函数。它们具有以下声明：
1. virtual void before_end_of_elaboration():
    在构建模块层次结构后调用
2. virtual void end_of_elaboration(): 
    在推导结束时调用，即在所有对before_end_of_elaboration的回调完成之后，并且在这些回调执行的任何实例化或端口绑定完成后，并在开始仿真之前调用。
3. virtual void start_of_simulation(): 
    a) 当应用程序首次为sc_start调用或在仿真的最初启动时立即调用，如果仿真是在内核的直接触发的。
    b) 如果应用程序对sc_start多次调用，则在第一次调用sc_start时调用start_of_simulation()。 
    c) 在调用end_of_elaboration的回调之后，并在调用调度器的初始化阶段之前调用。
4. virtual void end_of_simulation():
    a) 当调度器因sc_stop而停止时调用，或者仿真在内核的直接控制下结束时调用。
    b) 即使sc_stop被多次调用，也只会调用一次。

```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(STAGE) {
  SC_CTOR(STAGE) { // elaboration
    std::cout << sc_time_stamp() << ": Elaboration: constructor" << std::endl;
    SC_THREAD(thread); // initialization + simulation
  }
  ~STAGE() { // cleanup
    std::cout << sc_time_stamp() << ": Cleanup: desctructor" << std::endl;
  }
  void thread() {
    std::cout << sc_time_stamp() << ": Execution.initialization" << std::endl;
    int i = 0;
    while(true) {
      wait(1, SC_SEC); // advance-time
      std::cout << sc_time_stamp() << ": Execution.simulation" << std::endl; // evaluation
      if (++i >= 2) {
        sc_stop(); // stop simulation after 2 iterations
      }
    }
  }
  void before_end_of_elaboration() {
    std::cout << "before end of elaboration" << std::endl;
  }
  void end_of_elaboration() {
    std::cout << "end of elaboration" << std::endl;
  }
  void start_of_simulation() {
    std::cout << "start of simulation" << std::endl;
  }
  void end_of_simulation() {
    std::cout << "end of simulation" << std::endl;
  }
};

int sc_main(int, char*[]) {
  STAGE stage("stage"); // Elaboration
  sc_start(); // Execution till sc_stop
  return 0; // Cleanup
}
```
上面代码输出结果:
***0 s: Elaboration: constructor
before end of elaboration
end of elaboration
start of simulation
0 s: Execution.initialization
1 s: Execution.simulation
2 s: Execution.simulation
Info: /OSCI/SystemC: Simulation stopped by user.
end of simulation
2 s: Cleanup: desctructor***
# Time Notation
让我们首先了解两种时间测量的区别：
1. wall-clock时间，执行开始到完成的时间，包括等待其他系统活动和应用程序的时间。
2. 仿真时间，被仿真模型所跟踪的时间，可能小于或大于模拟的墙钟时间。

在SystemC中，sc_time是模拟内核用于跟踪仿真时间的数据类型。它定义了几种时间单位：SC_SEC（秒）、SC_MS（毫秒）、SC_US（微秒）、SC_NS（纳秒）、SC_PS（皮秒）和SC_FS（飞秒）。每个后续的时间单位都是之前的1/1000。

sc_time对象可以用作赋值、算术和比较操作的操作数：
+ 乘法允许其中一个操作数为双精度浮点数
+ 除法允许除数为双精度浮点数

SC_ZERO_TIME：
一个表示零时间值的宏。在编写零时间值时，例如创建增量通知或增量超时，使用这个常量是一个好习惯。
可以使用sc_time_stamp()函数获取当前模拟时间。

```c++
#include <systemc>
using namespace sc_core;

int sc_main(int, char*[]) {
sc_core::sc_report_handler::set_actions( "/IEEE_Std_1666/deprecated",
                                           sc_core::SC_DO_NOTHING ); // suppress warning due to set_time_resolution
  sc_set_time_resolution(1, SC_FS); // deprecated function but still useful, default is 1 PS
  sc_set_default_time_unit(1, SC_SEC); // change time unit to 1 second
  std::cout << "1 SEC =     " << sc_time(1, SC_SEC).to_default_time_units() << " SEC"<< std::endl;
  std::cout << "1  MS = " << sc_time(1, SC_MS).to_default_time_units()  << " SEC"<< std::endl;
  std::cout << "1  US = " << sc_time(1, SC_US).to_default_time_units()  << " SEC"<< std::endl;
  std::cout << "1  NS = " << sc_time(1, SC_NS).to_default_time_units()  << " SEC"<< std::endl;
  std::cout << "1  PS = " << sc_time(1, SC_PS).to_default_time_units()  << " SEC"<< std::endl;
  std::cout << "1  FS = " << sc_time(1, SC_FS).to_default_time_units()  << " SEC"<< std::endl;
  sc_start(7261, SC_SEC); // run simulation for 7261 second
  double t = sc_time_stamp().to_seconds(); // get time in second
  std::cout << int(t) / 3600 << " hours, " << (int(t) % 3600) / 60 << " minutes, " << (int(t) % 60) << "seconds" << std::endl;
  return 0;
}
```

上面代码输出结果:
***1 SEC =     1 SEC
1  MS = 0.001 SEC
1  US = 1e-06 SEC
1  NS = 1e-09 SEC
1  PS = 1e-12 SEC
1  FS = 1e-15 SEC
2 hours, 1 minutes, 1seconds***
# Concurrency
updating