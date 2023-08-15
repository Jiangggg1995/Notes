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
SystemC应用程序的执行分为三个阶段：
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
SystemC使用模拟进程来模拟并发。但这不是真正的并发执行。
当多个进程被模拟为同时运行时，只有一个在特定的时间执行。接着，仿真时间保持不变，直到所有并发进程在当前仿真时间完成任务。
因此，这些进程在相同的“模拟时间”上并行运行。这与Go语言等的真正的并发不同。
让我们用一个简单的例子来理解模拟的并发。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(CONCURRENCY) {
  SC_CTOR(CONCURRENCY) { // constructor
    SC_THREAD(thread1); // register thread1
    SC_THREAD(thread2); // register thread2
  }
  void thread1() {
    while(true) { // infinite loop
      std::cout << sc_time_stamp() << ": thread1" << std::endl;
      wait(2, SC_SEC); // trigger again after 2 "simulated" seconds
    }
  }
  void thread2() {
    while(true) {
      std::cout << "\t" << sc_time_stamp() << ": thread2" << std::endl;
      wait(3, SC_SEC);
    }
  }
};

int sc_main(int, char*[]) {
  CONCURRENCY concur("concur"); // define an object
  sc_start(10, SC_SEC); // run simulation for 10 seconds
  return 0;
}
```
上面代码输出结果:
***0 s: thread1
        0 s: thread2
2 s: thread1
        3 s: thread2
4 s: thread1
        6 s: thread2
6 s: thread1
8 s: thread1
        9 s: thread2***
# Event
一个事件是一个类`sc_event`的对象，用于进程同步。 一个进程实例可以在事件发生时被触发或恢复。 任何给定的事件可以在许多不同的情况下被通知。

sc_event有以下方法：
1. `void notify()`: 创建一个立即通知
2. `void notify(const sc_time&), void notify(double, sc_time_unit)`: 
    a) 零时间：创建一个delta通知。 
    b) 非零时间：在给定的时间创建一个定时通知，这个定时通知是相对于调用notify函数时刻的仿真时间。
3. cancel(): 删除此事件的任何待处理通知 
    a) 对于任何给定的事件，最多只能有一个待处理通知存在。 
    b) 即时通知不能被取消。

约束条件：

1. 类sc_event的对象可以在推导或仿真过程中构造。
2. 可以在推导或仿真过程中通知事件，但在以下情况下创建立即通知是错误的： 
    a) before_end_of_elaboration, 
    b) end_of_elaboration,
    c) start_of_simulation.

一个给定的事件不应该超过一个待处理的通知：
1. 如果已经有一个通知挂起的事件发生notify函数的调用，只有计划在最早时间发生的通知会存活。
2. 计划在后面时间发生的通知将被取消（或者从未被安排）。
3. 立即通知比delta通知早发生，delta通知又比定时通知早发生。这不考虑以何种顺序调用notify函数。
4. 
事件可以相互结合，并与定时器结合使用。这个例子显示了一个进程只等待一个事件的情况。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(EVENT) {
  sc_event e; // declare an event
  SC_CTOR(EVENT) {
    SC_THREAD(trigger); //register a trigger process
    SC_THREAD(catcher); // register a catcher process
  }
  void trigger() {
    while (true) { // infinite loop
      e.notify(1, SC_SEC); // trigger after 1 second
      if (sc_time_stamp() == sc_time(4, SC_SEC)) {
        e.cancel(); // cancel the event triggered at time = 4 s
      }
      wait(2, SC_SEC); // wait for 2 seconds before triggering again
    }
  }
  void catcher() {
    while (true) { // loop forever
      wait(e); // wait for event
      std::cout << "Event cateched at " << sc_time_stamp() << std::endl; // print to console
    }
  }
};

int sc_main(int, char*[]) {
  EVENT event("event"); // define object
  sc_start(8, SC_SEC); // run simulation for 8 seconds
  return 0;
}
```
上面代码输出结果:
***Event cateched at 1 s
Event cateched at 3 s
Event cateched at 7 s***
# Combined Events
SystemC支持以下形式的wait()函数：
1. wait(): 等待敏感列表中的事件（SystemC 1.0）。
2. wait(e1): 等待事件e1。
3. wait(e1 | e2 | e3): 等待事件e1、e2或e3。
4. wait(e1 & e2 & e3): 等待事件e1、e2和e3。
5. wait(200, SC_NS): 等待200纳秒。
6. wait(200, SC_NS, e1): 在200纳秒后等待事件e1。
7. wait(200, SC_NS, e1 | e2 | e3): 在200纳秒后等待事件e1、e2或e3。
8. wait(200, SC_NS, e1 & e2 & e3): 在200纳秒后等待事件e1、e2和e3。
9. wait(sc_time(200, SC_NS)): 等待200纳秒。
10. wait(sc_time(200, SC_NS), e1): 在200纳秒后等待事件e1。
11. wait(sc_time(200, SC_NS), e1 | e2 | e3): 在200纳秒后等待事件e1、e2或e3。
12. wait(sc_time(200, SC_NS), e1 & e2 & e3 ): 在200纳秒后等待事件e1、e2和e3。
13. wait(200): 等待200个时钟周期，仅限SC_CTHREAD（SystemC 1.0）。
14. wait(0, SC_NS): 等待一个delta周期。
15. wait(SC_ZERO_TIME): 等待一个delta周期。

注意： 在SystemC 2.0中，混合使用"|"运算符和"&"运算符是不支持的。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(COMBINED) {
  sc_event e1, e2, e3, e4, e5, e6, e7, e8, e9, e10; // declare multiple events
  SC_CTOR(COMBINED) {
    SC_THREAD(trigger); // reigster trigger
    SC_THREAD(catcher_0); // register catchers
    SC_THREAD(catcher_1);
    SC_THREAD(catcher_2and3);
    SC_THREAD(catcher_4or5);
    SC_THREAD(catcher_timeout_or_6);
    SC_THREAD(catcher_timeout_or_7or8);
    SC_THREAD(catcher_timeout_or_9and10);
  }
  void trigger(void) {
    e1.notify(1, SC_SEC);  // e1 fires at 1s
    e2.notify(2, SC_SEC);  // ...
    e3.notify(3, SC_SEC);
    e4.notify(4, SC_SEC);
    e5.notify(5, SC_SEC);
    e6.notify(6, SC_SEC);
    e7.notify(7, SC_SEC);
    e8.notify(8, SC_SEC);
    e9.notify(9, SC_SEC);
    e10.notify(10, SC_SEC); // e10 fires at 10s
  }
  void catcher_0(void) {
    wait(2, SC_SEC); // timer triggered
    std::cout << sc_time_stamp() << ": 2sec timeout" << std::endl;
  }
  void catcher_1(void) {
    wait(e1); // e1 triggered
    std::cout << sc_time_stamp() << ": catch e1" << std::endl;
  }
  void catcher_2and3(void) {
    wait(e2 & e3); // e2 and e3
    std::cout << sc_time_stamp() << ": catch e2 and e3" << std::endl;
  }
  void catcher_4or5(void) {
    wait(e4 | e5); // e4 or e5
    std::cout << sc_time_stamp() << ": catch e4 or e5" << std::endl;
  }
  void catcher_timeout_or_6(void) {
    wait(sc_time(5, SC_SEC), e6); // timer or e6
    std::cout << sc_time_stamp() << ": 5sec timeout or catch e6"<< std::endl;
  }
  void catcher_timeout_or_7or8(void) {
    wait(sc_time(20, SC_SEC), e7 | e8); // timer or (e7 or e8)
    std::cout << sc_time_stamp() << ": 20sec timeout or catch e7 or e8" << std::endl;
  }
  void catcher_timeout_or_9and10(void) {
    wait(sc_time(20, SC_SEC), e9 & e10); // timer or (e9 and e10)
    std::cout << sc_time_stamp() << ": 20sec timeout or catch (e9 and e10)" << std::endl;
  }
};

int sc_main(int, char*[]) {
  COMBINED combined("combined");
  sc_start();
  return 0;
}
```
上面代码输出结果:
***1 s: catch e1
2 s: 2sec timeout
3 s: catch e2 and e3
4 s: catch e4 or e5
5 s: 5sec timeout or catch e6
7 s: 20sec timeout or catch e7 or e8
10 s: 20sec timeout or catch (e9 and e10)***
# Delta Cycle
Delta周期可以被看作是仿真中一个非常小的步骤的时间，这不会增加用户可见的时间。 
Delta周期由单独的评估和更新阶段组成，在特定模拟时间可能发生多个delta周期。 当信号赋值发生时，其他进程在下一个delta周期之前看不到新分配的值。

Delta周期的使用场景包括：
1. notify(SC_ZERO_TIME)会在下一个delta周期的评估阶段中触发事件通知，这被称为"delta通知"。
2. 对request_update()的直接或间接调用会导致在当前delta周期的更新阶段中调用update()方法。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(DELTA) {
  int x = 1, y = 1; // defines two member variables
  SC_CTOR(DELTA) {
    SC_THREAD(add_x); // x += 2
    SC_THREAD(multiply_x); // x *= 3
    SC_THREAD(add_y); // y += 2
    SC_THREAD(multiply_y); // y *= 3
  }
  void add_x() { // x += 2 happens first
    std::cout << "add_x: " << x << " + 2" << " = ";
    x += 2;
    std::cout << x << std::endl;
  }
  void multiply_x() { // x *= 3 happens after a delta cycle
    wait(SC_ZERO_TIME);
    std::cout << "multiply_x: " << x << " * 3" << " = ";
    x *= 3;
    std::cout << x << std::endl;
  }
  void add_y() { // y += 2 happens after a delta cycle
    wait(SC_ZERO_TIME);
    std::cout << "add_y: " << y << " + 2" << " = ";
    y += 2;
    std::cout << y << std::endl;
  }
  void multiply_y() { // y *=3 happens first
    std::cout << "multiply_y: " << y << " * 3" << " = ";
    y *= 3;
    std::cout << y << std::endl;
  }
};

int sc_main(int, char*[]) {
  DELTA delta("delta");
  sc_start();
  return 0;
}
```
上面代码输出结果:
***add_x: 1 + 2 = 3
multiply_y: 1 * 3 = 3
add_y: 3 + 2 = 5
multiply_x: 3 * 3 = 9***
# Sensitivity
一个进程实例的敏感性是指可能导致进程被恢复或触发的一系列事件和超时。 如果一个事件已经被添加到进程实例的静态敏感性或动态敏感性中，那么该进程实例就会对该事件敏感。 超时发生在给定的时间间隔过去之后。

有两种类型的敏感性：
1. 静态敏感性在推导过程中被固定，每个模块中的每个进程都支持使用敏感性列表进行支持。
2. 动态敏感性可能会随着时间的变化而变化，由进程本身控制，可以使用wait()函数对线程进行支持，或者使用next_trigger()函数对方法进行支持。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(SENSITIVITY) {
  sc_event e1, e2; // events for inter-process triggering
  SC_CTOR(SENSITIVITY) {
    SC_THREAD(trigger_1); // register processes
    SC_THREAD(trigger_2);
    SC_THREAD(catch_1or2_dyn);
    SC_THREAD(catch_1or2_static);
    sensitive << e1 << e2; // static sensitivity for the preceeding process, can only "OR" the triggers
  }
  void trigger_1() {
    wait(SC_ZERO_TIME); // delay trigger by a delta cycle, make sure catcher is ready
    while (true) {
      e1.notify(); // trigger e1
      wait(2, SC_SEC); // dynamic sensitivity, re-trigger after 2 s
    }
  }
  void trigger_2() { // delay trigger by a delta cycle
    wait(SC_ZERO_TIME);
    while (true) {
      e2.notify(); // trigger e2
      wait(3, SC_SEC); // dynamic sensitivity, re-trigger after 3 s
    }
  }
  void catch_1or2_dyn() {
    while (true) {
      wait(e1 | e2); // dynamic sensitivity
      std::cout << "Dynamic sensitivty: e1 or e2 @ " << sc_time_stamp() << std::endl;
    }
  }
  void catch_1or2_static(void) {
    while (true) {
      wait(); // static sensitivity
      std::cout << "Static sensitivity: e1 or e2 @ " << sc_time_stamp() << std::endl;
    }
  }
};

int sc_main(int, char*[]) {
  SENSITIVITY sensitivity("sensitivity");
  sc_start(7, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***Static sensitivity: e1 or e2 @ 0 s
Dynamic sensitivty: e1 or e2 @ 0 s
Static sensitivity: e1 or e2 @ 2 s
Dynamic sensitivty: e1 or e2 @ 2 s
Static sensitivity: e1 or e2 @ 3 s
Dynamic sensitivty: e1 or e2 @ 3 s
Static sensitivity: e1 or e2 @ 4 s
Dynamic sensitivty: e1 or e2 @ 4 s
Static sensitivity: e1 or e2 @ 6 s
Dynamic sensitivty: e1 or e2 @ 6 s***
# Initialization
初始化是执行阶段的一部分，在sc_start()之后发生。在初始化期间，它会按给定的顺序执行以下三个步骤：
1. 运行更新阶段，但不继续到delta通知阶段。
2. 将对象层次结构中的method和thread进程实例添加到可运行进程的集合中，排除以下两种情况：
        a) 对于那些已经调用了dont_initialize函数的进程实例；
        b) 到达时间的thread进程。
3. 运行delta通知阶段。在delta通知阶段的结束时，转到评估阶段。

注意：
1. 更新和delta通知阶段是必要的，因为可以在推导过程中创建更新请求，以设置原始通道的初始值，例如从类sc_inout的initialize函数中。
2. 在SystemC 1.0中，
    a) 在模拟的初始化阶段中不执行线程进程。 
    b) 如果方法进程对输入信号/端口敏感，则在模拟的初始化阶段中执行方法进程。
3. SystemC 2.0调度程序将在模拟的初始化阶段中执行所有thread进程和method进程。 如果thread进程的行为在SystemC 1.0和SystemC 2.0之间有所不同，则在thread进程的无限循环之前插入一个wait()语句。
4. 在初始化阶段，进程（SystemC 1.0中的SC_METHOD；SystemC 2.0中的SC_METHOD和SC_THREAD）以未指定的顺序执行。
5. dont_initialize(): 用于防止调度程序在初始化阶段执行线程或方法进程。适用于调用该函数前最后声明的进程。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(INITIALIZATION) {
  sc_event e; // event for inter-process trigger
  SC_CTOR(INITIALIZATION) {
    SC_THREAD(trigger); // no static sensitivity
    SC_THREAD(catcher_1); // no static sensitivity
    SC_THREAD(catcher_2); // no static sensitivity
    SC_THREAD(catcher_3);
    sensitive << e; // statically sensitive to e
    dont_initialize(); // don't initialize
  }
  void trigger() {
    while (true) { // e triggered at 1, 3, 5, 7 ...
      e.notify(1, SC_SEC); // notify after 1 s
      wait(2, SC_SEC); // trigger every 2 s
    }
  }
  void catcher_1() {
    while (true) {
      std::cout << sc_time_stamp() << ": catcher_1 triggered" << std::endl;
      wait(e); // dynamic sensitivity
    }
  }
  void catcher_2() {
    wait(e); // avoid initialization --- mimic systemC 1.0 behavior
    while (true) {
      std::cout << sc_time_stamp() << ": catcher_2 triggered" << std::endl;
      wait(e); // dynamic sensitivity
    }
  }
  void catcher_3() { // relies on dont_initialize() to avoid initialization
    while (true) {
      std::cout << sc_time_stamp() << ": catcher_3 triggered" << std::endl;
      wait(e); // dynamic sensitivity
    }
  }
};

int sc_main(int, char*[]) {
  INITIALIZATION init("init");
  sc_start(4, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: catcher_1 triggered
1 s: catcher_3 triggered
1 s: catcher_1 triggered
1 s: catcher_2 triggered
3 s: catcher_3 triggered
3 s: catcher_2 triggered
3 s: catcher_1 triggered***
# Process: Method
Method:
1. 可能有静态敏感性。
2. 只有method进程可以调用函数next_trigger来创建动态敏感性。
3. 由于进程自身执行的立即通知，无论如何method进程实例的静态敏感性或动态敏感性，都不能使其可作为可运行对象。

next_trigger():
1. 是类sc_module的成员函数，是类sc_prim_channel的成员函数，也是非成员函数。
2. 可以从以下位置调用： a) 模块自身的成员函数； b) 通道的成员函数，或者 c) 从任何最终从method进程中调用的函数。

注意：
1. 在method进程返回时，声明的任何局部变量都将被销毁。由method进程处理的模块数据成员应当被持久化保存。

回忆SC_METHOD和SC_THREAD之间的区别：
1. SC_METHOD(func): 没有自己的执行线程，不消耗模拟时间，不能被挂起，也不能调用调用wait()的代码。
2. SC_THREAD(func): 有自己的执行线程，可能消耗模拟时间，可以被挂起，可以调用wait()。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(PROCESS) {
  SC_CTOR(PROCESS) { // constructor
    SC_THREAD(thread); // register a thread process
    SC_METHOD(method); // register a method process
  }
  void thread() {
    int idx = 0; // declare only once
    while (true) { // loop forever
      std::cout << "thread"<< idx++ << " @ " << sc_time_stamp() << std::endl;
      wait(1, SC_SEC); // re-trigger after 1 s
    }
  }
  void method() {
    // notice there's no while loop here
    int idx = 0; // re-declare every time method is triggered
    std::cout << "method" << idx++ << " @ " << sc_time_stamp() << std::endl;
    next_trigger(1, SC_SEC);
  }
};

int sc_main(int, char*[]) {
  PROCESS process("process");
  sc_start(4, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***method0 @ 0 s
thread0 @ 0 s
method0 @ 1 s
thread1 @ 1 s
method0 @ 2 s
thread2 @ 2 s
method0 @ 3 s
thread3 @ 3 s***
# Event Queue
这是一个关于事件队列的描述。事件队列有以下特点：
1. 具有一个成员函数 notify()，类似于事件；
2. 是一个体系通道，可以有多个待处理的通知，这与事件只能安排一个未解决通知不同；
3. 只能在推导过程中构造；
4. 不支持立即通知。

成员函数包括：
1. void notify(double, sc_time_unit) 或 void notify(const sc_time&)： a) 零时间，即 SC_ZERO_TIME：表示一个增量通知； b) 非零时间：表示相对于调用 notify 函数时的模拟时间安排的通知。
2. void cancel_all()：立即删除此事件队列对象的所有待处理通知，包括增量和定时通知。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(QUEUE) {
  sc_event e;
  sc_event_queue eq;
  SC_CTOR(QUEUE) {
    SC_THREAD(trigger);
    SC_THREAD(catch_e);
    sensitive << e; // catch_e() will be triggered by event e
    dont_initialize(); // don't run cach_e() during initialization phase
    SC_THREAD(catch_eq);
    sensitive << eq; // cach_eq() will be triggered by event queue eq
    dont_initialize(); // don't run catch_eq() during initialization phase
  }
  void trigger() {
    while (true) {
      e.notify(2, SC_SEC); // trigger e afer 2 s
      e.notify(1, SC_SEC); // trigger e after 1 s, replaces previous trigger
      eq.notify(2, SC_SEC); // trigger eq after 2 s
      eq.notify(1, SC_SEC); // trigger eq after 1 s, both triggers available
      wait(10, SC_SEC); // another round
    }
  }
  void catch_e() {
    while (true) {
      std::cout << sc_time_stamp() << ": catches e" << std::endl;
      wait(); // no parameter --> wait for static sensitivity, i.e. e
    }
  }
  void catch_eq() {
    while (true) {
      std::cout << sc_time_stamp() << ": catches eq" << std::endl;
      wait(); // wait for eq
    }
  }
};

int sc_main(int, char*[]) {
  QUEUE queue("queue"); // instantiate object 
  sc_start(20, SC_SEC); // run simulation for 20 s
  return 0;
}
```
上面代码输出结果:
***1 s: catches e
1 s: catches eq
2 s: catches eq
11 s: catches e
11 s: catches eq
12 s: catches eq***
# Combined Event Queue
updating