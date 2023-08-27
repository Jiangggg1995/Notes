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
1. 多个事件队列可以通过"或"操作组合成进程的静态灵敏度。 在静态灵敏度中不能使用"与"操作。
2. Event Queue不能用作wait()函数的输入，因此不能用于动态灵敏度。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(QUEUE_COMBINED) {
  sc_event_queue eq1, eq2;
  SC_CTOR(QUEUE_COMBINED) {
    SC_THREAD(trigger);
    SC_THREAD(catcher);
    sensitive << eq1 << eq2; // eq1 "or" eq2, cannot "and"
    dont_initialize();
  }
  void trigger() {
    eq1.notify(1, SC_SEC); // eq1 at 1 s
    eq1.notify(2, SC_SEC); // eq1 at 2 s
    eq2.notify(2, SC_SEC); // eq2 at 2 s
    eq2.notify(3, SC_SEC); // eq2 at 3 s
  }
  void catcher() {
    while (true) {
      std::cout << sc_time_stamp() << ": catches trigger" << std::endl;
      wait(); // cannot use event queue in dynamic sensitivity
    }
  }
};

int sc_main(int, char*[]) {
  QUEUE_COMBINED combined("combined");
  sc_start();
  return 0;
}
```
上面代码输出结果:
***1 s: catches trigger
2 s: catches trigger
3 s: catches trigger***
# Mutex
互斥锁：
1. 是一个预定义的信道，用于模拟多个并发进程共享的资源上的互斥锁的行为。
2. 应处于以下两个独占状态之一：解锁或锁定： 
    a) 一个进程一次只能锁定给定的互斥锁。
    b) 互斥锁只能由锁定它的进程解锁。在解锁后，互斥锁可以被另一个进程锁定。

成员函数：
1. int lock(): 
    a) 如果互斥锁未锁定，lock()将锁定互斥锁并返回。 
    b) 如果互斥锁已锁定，lock()将挂起，直到互斥锁被另一个进程解锁。 
    c) 如果多个进程在同一周期尝试锁定互斥锁，那么哪个进程实例获得锁的选择将是不确定的。 
    d) 无条件返回值0。
2. int trylock(): 
    a) 如果互斥锁未锁定，trylock()将锁定互斥锁并返回值0。
    b) 如果互斥锁已锁定，trylock()将立即返回值–1。互斥锁保持锁定状态。
3. int unlock(): 
    a) 如果互斥锁未锁定，unlock()将返回值–1。互斥锁保持未锁定状态。 
    b) 如果互斥锁是由调用进程之外的其他进程实例锁定的，unlock()将返回值–1。互斥锁保持锁定状态。 
    c) 如果互斥锁是由调用进程锁定的，member函数unlock()将解锁互斥锁并返回值0。
解锁互斥锁后会向其他进程发出即时通知。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(MUTEX) {
  sc_mutex m;
  SC_CTOR(MUTEX) {
    SC_THREAD(thread_1);
    SC_THREAD(thread_2);
  }
  void thread_1() {
    while (true) {
      if (m.trylock() == -1) { // try to lock the mutex
        m.lock(); // failed, wait to lock
        std::cout << sc_time_stamp() << ": thread_1 obtained resource by lock()" << std::endl;
      } else { // succeeded
        std::cout << sc_time_stamp() << ": thread_1 obtained resource by trylock()" << std::endl;
      }
      wait(1, SC_SEC); // occupy mutex for 1 s
      m.unlock(); // unlock mutex
      std::cout << sc_time_stamp() << ": unlocked by thread_1" << std::endl;
      wait(SC_ZERO_TIME); // give time for the other process to lock the mutex
    }
  }
  void thread_2() {
    while (true) {
      if (m.trylock() == -1) { // try to lock the mutex
        m.lock(); // failed, wait to lock
        std::cout << sc_time_stamp() << ": thread_2 obtained resource by lock()" << std::endl;
      } else { // succeeded
        std::cout << sc_time_stamp() << ": thread_2 obtained resource by trylock()" << std::endl;
      }
      wait(1, SC_SEC); // occupy mutex for 1 s
      m.unlock(); // unlock mutex
      std::cout << sc_time_stamp() << ": unlocked by thread_2" << std::endl;
      wait(SC_ZERO_TIME); // give time for the other process to lock the mutex
    }
  }
};

int sc_main(int, char*[]) {
  MUTEX mutex("mutex");
  sc_start(4, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: thread_1 obtained resource by trylock()
1 s: unlocked by thread_1
1 s: thread_2 obtained resource by lock()
2 s: unlocked by thread_2
2 s: thread_1 obtained resource by lock()
3 s: unlocked by thread_1
3 s: thread_2 obtained resource by lock()***
# Semaphore
信号量：
1. 是一个预定义的信道，用于模拟软件信号量的行为，以提供对共享资源的有限并发访问。
2. 具有整数值，称为信号量值，在构造信号量时设置为允许的并发访问数量。如果信号量初始值为1，则信号量等同于互斥锁。

成员函数：
1. int wait(): 
    a) 如果信号量值大于0，wait()将减小信号量值并返回。 
    b) 如果信号量值等于0，wait()将挂起，直到信号量值被其他进程增加（通过另一个进程）。 
    c) 无条件返回值0。
2. int trywait(): 
   a) 如果信号量值大于0，trywait()将减小信号量值并返回值0。 
   b) 如果信号量值等于0，trywait()将立即返回值–1，而不修改信号量值。
3. int post(): 
   a) 将增加信号量值。 
   b) 将使用立即通知向任何等待的进程发出信号，表示增加信号量值的行为。 
   c) 无条件返回值0。
4. int get_value(): 
    a) 应返回信号量值。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(SEMAPHORE) {
  sc_semaphore s; // declares semaphore
  SC_CTOR(SEMAPHORE) : s(2) { // init semaphore with 2 resources
    SC_THREAD(thread_1); // register 3 threads competing for resources
    SC_THREAD(thread_2);
    SC_THREAD(thread_3);
  }
  void thread_1() {
    while (true) {
      if (s.trywait() == -1) { // try to obtain a resource
        s.wait(); // if not successful, wait till resource is available
      }
      std::cout<< sc_time_stamp() << ": locked by thread_1, value is " << s.get_value() << std::endl;
      wait(1, SC_SEC); // occupy resource for 1 s
      s.post(); // release resource
      std::cout<< sc_time_stamp() << ": unlocked by thread_1, value is " << s.get_value() << std::endl;
      wait(SC_ZERO_TIME); // give time for the other process to lock
    }
  }
  void thread_2() {
    while (true) {
      if (s.trywait() == -1) { // try to obtain a resource
        s.wait(); // if not successful, wait till resource is available
      }
      std::cout<< sc_time_stamp() << ": locked by thread_2, value is " << s.get_value() << std::endl;
      wait(1, SC_SEC); // occupy resource for 1 s
      s.post(); // release resource
      std::cout<< sc_time_stamp() << ": unlocked by thread_2, value is " << s.get_value() << std::endl;
      wait(SC_ZERO_TIME); // give time for the other process to lock
    }
  }
  void thread_3() {
    while (true) {
      if (s.trywait() == -1) { // try to obtain a resource
        s.wait(); // if not successful, wait till resource is available
      }
      std::cout<< sc_time_stamp() << ": locked by thread_3, value is " << s.get_value() << std::endl;
      wait(1, SC_SEC); // occupy resource for 1 s
      s.post(); // release resource
      std::cout<< sc_time_stamp() << ": unlocked by thread_3, value is " << s.get_value() << std::endl;
      wait(SC_ZERO_TIME); // give time for the other process to lock
    }
  }
};

int sc_main(int, char*[]) {
  SEMAPHORE semaphore("semaphore");
  sc_start(4, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: locked by thread_1, value is 1
0 s: locked by thread_2, value is 0
1 s: unlocked by thread_1, value is 1
1 s: unlocked by thread_2, value is 2
1 s: locked by thread_3, value is 1
1 s: locked by thread_2, value is 0
2 s: unlocked by thread_3, value is 1
2 s: unlocked by thread_2, value is 2
2 s: locked by thread_1, value is 1
2 s: locked by thread_2, value is 0
3 s: unlocked by thread_1, value is 1
3 s: unlocked by thread_2, value is 2
3 s: locked by thread_3, value is 1
3 s: locked by thread_2, value is 0***
# FIFO
sc_fifo是一个预定义的原始信道，用于模拟FIFO（先进先出）缓冲区的行为。它具有固定数量容积来存储数据。它实现了sc_fifo_in_if<\T> 接口和the sc_fifo_out_if<\T> 接口。

构造函数：
1. explicit sc_fifo(int size_ = 16): 调用基类的构造函数。
2. explicit sc_fifo(const char* name_, int size_ = 16): 调用基类的构造函数，使用给定的名称初始化字符串。
两个构造函数都通过size_参数初始固定容量。这个容量的size应该大于0。

成员函数：
1. void read(T&), T read(): 
    a) 返回最近写入fifo的值，并从fifo中移除该值，使其无法再次读取。 
    b) 从fifo中读取值的顺序应与写入fifo的顺序完全匹配。 
    c) 在当前delta周期中写入fifo的值在该周期内不可读，但会在下一个delta周期中变为可读。 
    d) 如果fifo为空，则读取事件挂起，直到被通知写入事件完成。
2. bool nb_read(T&): 
    a), b), c) 与read()相同 
    d) 如果fifo为空，nb_read()函数会立即返回，不修改fifo的状态，不调用request_update，并返回false。否则，如果可以从fifo中读取值，nb_read()的返回值为true。
3. operator T(): 等价于 "operator T() {return read();}"

成员函数：
1. write(const T&): 
    a) 将作为参数传递的值写入fifo。 
    b) 可以在单个delta周期内写入多个值。 
    c) 如果在同一delta周期中从fifo中读取了值，则在创建的空插槽中的值不会在下一个delta周期中变为可写。 
    d) 如果fifo已满，write()函数会在数据读取事件通知之前挂起。
2. bool nb_write(const T&): 
    a), b), c) 与write()相同 
    d) 如果fifo已满，nb_write()函数会立即返回，不修改fifo的状态，不调用request_update，并返回false。否则，nb_write()的返回值为true。
3. operator=: 等价于 "sc_fifo& operator= (const T& a) {write(a); return *this;}"

成员函数：
1. sc_event& data_written_event(): 应返回对数据写入事件的引用，该事件在delta通知阶段通知，即在写入值到fifo的delta周期结束时发生。
2. sc_event& data_read_event(): 应返回对数据读取事件的引用，该事件在delta通知阶段通知，即在从fifo中读取值的delta周期结束时发生。

成员函数：
1. int num_available(): 返回当前delta周期中可用于读取的值的数量。计算不包括在当前delta周期中读取的值，但不包括在当前delta周期中写入的值。
2. int num_free(): 返回当前delta周期中可用于写入的空闲容积的数量。计算不包括在当前delta周期中写入的容积，但不包括由读取在当前delta周期中产生的空闲容积。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(FIFO) {
  sc_fifo<int> f1, f2, f3;
  SC_CTOR(FIFO) : f1(2), f2(2), f3(2) { // fifo with size 2
    SC_THREAD(generator1);
    SC_THREAD(consumer1);

    SC_THREAD(generator2);
    SC_THREAD(consumer2);

    SC_THREAD(generator3);
    SC_THREAD(consumer3);
  }
  void generator1() { // blocking write
    int v = 0;
    while (true) {
      f1.write(v); // same as f = v, which is not recommended.
      std::cout << sc_time_stamp() << ": generator1 writes " << v++ << std::endl;
      wait(1, SC_SEC); // write every 1 s
    }
  }
  void consumer1() { // blocking read
    int v = -1;
    while (true) {
      f1.read(v); // same as v = int(f), which is not recommended; or, v = f1.read();
      std::cout << sc_time_stamp() << ": consumer1 reads " << v << std::endl;
      wait(3, SC_SEC); // read every 3 s, fifo will fill up soon
    }
  }
  void generator2() { // non-blocking write
    int v = 0;
    while (true) {
      while (f2.nb_write(v) == false ) { // nb write until succeeded
        wait(f2.data_read_event()); // if not successful, wait for data read (a fifo slot becomes available)
      }
      std::cout << sc_time_stamp() << ": generator2 writes " << v++ << std::endl;
      wait(1, SC_SEC); // write every 1 s
    }
  }
  void consumer2() { // non-blocking read
    int v = -1;
    while (true) {
      while (f2.nb_read(v) == false) {
        wait(f2.data_written_event());
      }
      std::cout << sc_time_stamp() << ": consumer2 reads " << v << std::endl;
      wait(3, SC_SEC); // read every 3 s, fifo will fill up soon
    }
  }
  void generator3() { // free/available slots before/after write
    int v = 0;
    while (true) {
      std::cout << sc_time_stamp() << ": generator3, before write, #free/#available=" << f3.num_free() << "/" << f3.num_available() << std::endl;
      f3.write(v++);
      std::cout << sc_time_stamp() << ": generator3, after write, #free/#available=" << f3.num_free() << "/" << f3.num_available() << std::endl;
      wait(1, SC_SEC);
    }
  }
  void consumer3() { // free/available slots before/after read
    int v = -1;
    while (true) {
      std::cout << sc_time_stamp() << ": consumer3, before read, #free/#available=" << f3.num_free() << "/" << f3.num_available() << std::endl;
      f3.read(v);
      std::cout << sc_time_stamp() << ": consumer3, after read, #free/#available=" << f3.num_free() << "/" << f3.num_available() << std::endl;
      wait(3, SC_SEC); // read every 3 s, fifo will fill up soon
    }
  }
};

int sc_main(int, char*[]) {
  FIFO fifo("fifo");
  sc_start(10, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: generator1 writes 0
0 s: generator2 writes 0
0 s: generator3, before write, #free/#available=2/0
0 s: generator3, after write, #free/#available=1/0
0 s: consumer3, before read, #free/#available=1/0
0 s: consumer1 reads 0
0 s: consumer2 reads 0
0 s: consumer3, after read, #free/#available=1/0
1 s: generator1 writes 1
1 s: generator2 writes 1
1 s: generator3, before write, #free/#available=2/0
1 s: generator3, after write, #free/#available=1/0
2 s: generator1 writes 2
2 s: generator2 writes 2
2 s: generator3, before write, #free/#available=1/1
2 s: generator3, after write, #free/#available=0/1
3 s: consumer3, before read, #free/#available=0/2
3 s: consumer3, after read, #free/#available=0/1
3 s: generator3, before write, #free/#available=0/1
3 s: consumer1 reads 1
3 s: consumer2 reads 1
3 s: generator3, after write, #free/#available=0/1
3 s: generator1 writes 3
3 s: generator2 writes 3
4 s: generator3, before write, #free/#available=0/2
6 s: consumer1 reads 2
6 s: consumer3, before read, #free/#available=0/2
6 s: consumer3, after read, #free/#available=0/1
6 s: consumer2 reads 2
6 s: generator1 writes 4
6 s: generator3, after write, #free/#available=0/1
6 s: generator2 writes 4
7 s: generator3, before write, #free/#available=0/2
9 s: consumer3, before read, #free/#available=0/2
9 s: consumer3, after read, #free/#available=0/1
9 s: consumer1 reads 3
9 s: consumer2 reads 3
9 s: generator3, after write, #free/#available=0/1
9 s: generator1 writes 5
9 s: generator2 writes 5***
# Signal: read and write
sc_signal:
1. 是一个预定义的原始信道，用于模拟携带数电信号的线路的行为。
2. 它使用evaluate-update方案来确保在同时进行读取和写入操作时的行为是确定的。我们维护一个当前值和新值。
3. 它的write()方法将在新值与当前值不同时提交更新请求。
4. 它实现了sc_signal_inout_if接口。

构造函数：
1. `sc_signal()`: 
    从其初始化器列表中调用基类的构造函数，如下所示`sc_prim_channel(sc_gen_unique_name("signal"))`
3. `sc_signal(const char* name_)`: 
    从其初始化器列表中调用基类的构造函数，如下所示：`sc_prim_channel(name_)`

成员函数：
1. T& read() 或 operator const T&(): 
    返回信号的当前值的引用，但不应修改信号的状态。
2. void write(const T&): 
    修改信号的值，使信号在下个delta周期中具有新值（如成员函数read返回的值），但在此之前不变。
3. operator=: 
    等同于write()。
4. sc_event& default_event(), sc_event& value_changed_event(): 
    返回值更改事件的引用。
5. bool event(): 
    如果且仅当在前一个delta周期的更新阶段以及当前仿真时间信号的值发生了改变，则返回true。

与fifo相比：
1. sc_signal只有一个容量用于读写
2. sc_signal只在新值与当前值不同时触发更新请求
3. 从sc_signal中读取不会移除值

除了执行阶段外，sc_signal:
1. 可以在推导期间编写以初始化信号的值。
2. 可以在推导期间或仿真暂停时从函数sc_main编写，即在调用函数sc_start之前或之后。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(SIGNAL) {
  sc_signal<int> s;
  SC_CTOR(SIGNAL) {
    SC_THREAD(readwrite);
  }
  void readwrite() {
    s.write(3);
    std::cout << "s = " << s << "; " << s.read() << std::endl;
    wait(SC_ZERO_TIME);
    std::cout << "after delta_cycle, s = " << s << std::endl;
    
    s = 4;
    s = 5;
    int tmp = s;
    std::cout << "s = " << tmp << std::endl;
    wait(SC_ZERO_TIME);
    std::cout << "after delta_cycle, s = " << s.read() << std::endl;
  }
};

int sc_main(int, char*[]) {
  SIGNAL signal("signal");
  signal.s = -1;
  sc_start();
  return 0;
}
```
上面代码输出结果:
***s = -1; -1
after delta_cycle, s = 3
s = 3
after delta_cycle, s = 5***
# Signal: detect event
1. `sc_event& default_event(), sc_event& value_changed_event()`: 
    返回对值更改事件的引用。
2. `bool event()`: 此函数在以下情况下返回true：
    仅当在前一个delta周期的更新阶段以及当前仿真时间中信号的值发生变化时，才返回true。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(SIGNAL_EVENT) {
  sc_signal<int> s1, s2; // defines two signal channels
  SC_CTOR(SIGNAL_EVENT) {
    SC_THREAD(producer1);
    SC_THREAD(producer2);
    SC_THREAD(consumer); // consumer sensitive to (s1 OR s2)
    sensitive << s1 << s2; // same as: sensitive << s1.default_event() << s2.value_changed_event();
    dont_initialize();
  }
  void producer1() {
    int v = 1;
    while (true) {
      s1.write(v++); // write to s1
      wait(2, SC_SEC);
    }
  }
  void producer2() {
    int v = 1;
    while (true) {
      s2 = v++; // write to s2
      wait(3, SC_SEC);
    }
  }
  void consumer() {
    while (true) {
      if ( s1.event() == true && s2.event() == true) { // both triggered
        std::cout << sc_time_stamp() << ": s1 & s2 triggered" << std::endl; 
      } else if (s1.event() == true) { // only s1 triggered
        std::cout << sc_time_stamp() << ": s1 triggered" << std::endl; 
      } else { // only s2 triggered
        std::cout << sc_time_stamp() << ": s2 triggered" << std::endl; 
      }
      wait();
    }
  }
};

int sc_main(int, char*[]) {
  SIGNAL_EVENT signal_event("signal_event");
  sc_start(7, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: s1 & s2 triggered
2 s: s1 triggered
3 s: s2 triggered
4 s: s1 triggered
6 s: s1 & s2 triggered***
# Signal: many writers
sc_signal的类定义：
`template <class T, sc_writer_policy WRITER_POLICY = SC_ONE_WRITER> class sc_signal: public sc_signal_inout_if<T>, public sc_prim_channel {}`

1. 如果 WRITER_POLICY == SC_ONE_WRITER，则在仿真过程中的任何时候从多个进程实例写入给定的信号实例都应视为错误。
2. 如果 WRITER_POLICY == SC_MANY_WRITERS: 
    a) 在任何给定的评估阶段，从多个进程实例写入给定的信号实例都应视为错误。 
    b) 但不同的进程实例可能在不同的delta周期中写入给定的信号实例。
因此，默认情况下，一个sc_signal只有一个写入者；当声明为MANY_WRITERS时，多个写入者可以在不同时间写入信号通道。
至于消费者，一个sc_signal可以有多个消费者。它们可以在同一时间或不同时间从信号通道读取。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(MULTI) {
  sc_signal<int> s1; // a single-writer signal
  sc_signal<int, SC_MANY_WRITERS> s2; // a multi-writer signal
  SC_CTOR(MULTI) {
    SC_THREAD(writer1); // writes to s1
    SC_THREAD(writer2); // writes to s1 and s2
    SC_THREAD(consumer1);
    sensitive << s1; // sensitive to s1
    dont_initialize();
    SC_THREAD(consumer2);
    sensitive << s1 << s2; // sensitive to s1 and s2
    dont_initialize();
  }
  void writer1() {
    int v = 1; // init value
    while (true) {
      s1.write(v); // write to s1
      s2.write(v); // write to s2
      std::cout << sc_time_stamp() << ": writer1 writes " << v++ << std::endl;
      wait(1, SC_SEC); // write every 1 s
    }
  }
  void writer2() {
    int v = -1; // init value
    while (true) {
      // s1.write(v); /* cannot, otherwise runtime error: (E115) sc_signal<T> cannot have more than one driver*/
      wait(SC_ZERO_TIME); // needed to offset the write time. Otherwise runtime error: conflicting write in delta cycle 0 
      s2.write(v); // write to s2
      std::cout << sc_time_stamp() << ": writer2 writes " << v-- << std::endl;
      wait(1, SC_SEC); // write every 1 s
    }
  }
  void consumer1() {
    while (true) {
      std::cout << sc_time_stamp() << ": consumer1 reads s1=" << s1.read() << "; s2=" << s2.read() << std::endl; // read s1 and s2
      wait(); // wait for s1
    }
  }
  void consumer2() {
    while (true) {
      std::cout << sc_time_stamp() << ": consumer2 reads s1=" << s1.read() << "; s2=" << s2.read() << std::endl; // read s1 and s2
      wait(); // wait for s1 or s2
    }
  }
};

int sc_main(int, char*[]) {
  MULTI consumers("consumers");
  sc_start(2, SC_SEC); // run simulation for 2 s
  return 0;
}
```
上面代码输出结果:
***0 s: writer1 writes 1
0 s: consumer2 reads s1=1; s2=1
0 s: consumer1 reads s1=1; s2=1
0 s: writer2 writes -1
0 s: consumer2 reads s1=1; s2=-1
1 s: writer1 writes 2
1 s: consumer2 reads s1=2; s2=2
1 s: consumer1 reads s1=2; s2=2
1 s: writer2 writes -2
1 s: consumer2 reads s1=2; s2=-2***
# Resolved Signal
Resolved Signal是一个类sc_signal_resolved或类sc_signal_rv实例化的对象。它与sc_signal的区别在于，Resolved Signal可能被多个进程写入，冲突的值在通道内解析。
1. sc_signal_resolved是从类sc_signal派生出来的预定义的原语通道。
2. sc_signal_rv也是从类sc_signal派生出来的预定义的原语通道。
    a) sc_signal_rv与sc_signal_resolved相似。
    b) 不同的是，基类模板sc_signal的参数类型为sc_dt::sc_lv而不是sc_dt::sc_logic。

类定义：
1. `class sc_signal_resolved: public sc_signal<sc_dt::sc_logic,SC_MANY_WRITERS>`
2. `template <int W> class sc_signal_rv: public sc_signal<sc_dt::sc_lv<W>,SC_MANY_WRITERS>`

针对sc_signal_resolved的解析表：
  | 0 | 1 | Z | X |
0 | 0 | X | 0 | X |
1 | X | 1 | 1 | X |
Z | 0 | 1 | Z | X |
X | X | X | X | X |

简而言之，Resolved Signal通道可以同时被多个进程写入。这与只能每个delta周期被一个进程写入的sc_signal不同。
```c++
#include <systemc>
#include <vector> // use c++  vector lib
using namespace sc_core;
using namespace sc_dt; // sc_logic defined here
using std::vector; // use namespace for vector

SC_MODULE(RESOLVED_SIGNAL) {
  sc_signal_resolved rv; // a resolved signal channel
  vector<sc_logic> levels; // declares a vector of possible 4-level logic values
  SC_CTOR(RESOLVED_SIGNAL) : levels(vector<sc_logic>{sc_logic_0, sc_logic_1, sc_logic_Z, sc_logic_X}){ // init vector for possible 4-level logic values
    SC_THREAD(writer1);
    SC_THREAD(writer2);
    SC_THREAD(consumer);
  }
  void writer1() {
    int idx = 0;
    while (true) {
      rv.write(levels[idx++%4]); // 0,1,Z,X, 0,1,Z,X, 0,1,Z,X, 0,1,Z,X
      wait(1, SC_SEC); // writes every 1 s
    }
  }
  void writer2() {
    int idx = 0;
    while (true) {
      rv.write(levels[(idx++/4)%4]); // 0,0,0,0, 1,1,1,1, Z,Z,Z,Z, X,X,X,X
      wait(1, SC_SEC); // writes every 1 s
    }
  }
  void consumer() {
    wait(1, SC_SEC); // delay read by 1 s
    int idx = 0;
    while (true) {
      std::cout << " " << rv.read() << " |"; // print the read value (writer1 and writer2 resolved)
      if (++idx % 4 == 0) { std::cout << std::endl; } // print a new line every 4 values
      wait(1, SC_SEC); // read every 1 s
    }
  }
};

int sc_main(int, char*[]) {
  RESOLVED_SIGNAL resolved("resolved");
  sc_start(17, SC_SEC); // runs sufficient time to test all 16 resolve combinations
  return 0;
}
```
上面代码输出结果:
***0 | X | 0 | X |
X | 1 | 1 | X |
0 | 1 | Z | X |
X | X | X | X |***
# sc_signal<\bool>
sc_signal_in_if和sc_signal_in_if是提供适用于双值信号的额外成员函数的接口。 sc_signal实现了这些函数：
1. posedge_event()返回对事件的引用，该事件在通道值发生变化并且新通道值为true或'1'时通知。
2. negedge_event()返回对事件的引用，该事件在通道值发生变化并且新通道值为false或'0'时通知。
3. posedge()在当前仿真时间上，仅当在前一个delta周期的更新阶段中通道值发生变化，并且新通道值为true或'1'时才返回true。
4. negedge()在当前仿真时间上，仅当在前一个delta周期的更新阶段中通道值发生变化，并且新通道值为false或'0'时才返回true。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(SIGNAL_BOOL) {
  sc_signal<bool> b;
  SC_CTOR(SIGNAL_BOOL) {
    SC_THREAD(writer);
    SC_THREAD(consumer);
    sensitive << b; // triggered by every value change
    dont_initialize();
    SC_THREAD(consumer_pos);
    sensitive << b.posedge_event(); // triggered by value change to true
    dont_initialize();
    SC_THREAD(consumer_neg);
    sensitive << b.negedge_event(); // triggered by value change to false
    dont_initialize();
  }
  void writer() {
    bool v = true;
    while (true) {
      b.write(v); // write to channel
      v = !v; // toggle value
      wait(1, SC_SEC); // write every 1 s
    }
  }
  void consumer() {
    while (true) {
      if (b.posedge()) { // if new value is true
        std::cout << sc_time_stamp() << ": consumer receives posedge, b = " << b << std::endl;
      } else { // if new value is false
        std::cout << sc_time_stamp() << ": consumer receives negedge, b = " << b << std::endl;
      }
      wait(); // wait for any value change
    }
  }
  void consumer_pos() {
    while (true) {
      std::cout << sc_time_stamp() << ": consumer_pos receives posedge, b = " << b << std::endl;
      wait(); // wait for value change to true
    }
  }
  void consumer_neg() {
    while (true) {
      std::cout << sc_time_stamp() << ": consumer_neg receives negedge, b = " << b << std::endl;
      wait(); // wait for value change to false
    }
  }
};

int sc_main(int, char*[]) {
  SIGNAL_BOOL signal_bool("signal_bool");
  sc_start(4, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: consumer_pos receives posedge, b = 1
0 s: consumer receives posedge, b = 1
1 s: consumer_neg receives negedge, b = 0
1 s: consumer receives negedge, b = 0
2 s: consumer_pos receives posedge, b = 1
2 s: consumer receives posedge, b = 1
3 s: consumer_neg receives negedge, b = 0
3 s: consumer receives negedge, b = 0***
# Buffer
sc_buffer是从类sc_signal预定义的基本通道派生出来的原语通道。 
它与类sc_signal的不同之处在于，每当缓冲区被写入时都会通知值更改事件，而不仅仅是当信号的值发生变化时才通知。 
例如， 
如果"signal"的当前值\==1，写入1不会触发值更新事件。 
如果"buffer"的当前值\==1，写入1将触发值更新事件。
```C++
#include <systemc>
using namespace sc_core;

SC_MODULE(BUFFER) {
  sc_signal<int> s; // declares a signal channel
  sc_buffer<int> b; // declares a buffer channel
  SC_CTOR(BUFFER) {
    SC_THREAD(writer); // writes to both signal and buffer
    SC_THREAD(consumer1);
    sensitive << s; // triggered by signal
    dont_initialize();
    SC_THREAD(consumer2);
    sensitive << b; // triggered by buffer
    dont_initialize();
  }
  void writer() {
    int val = 1; // init value
    while (true) {
      for (int i = 0; i < 2; ++i) { // write same value to channel twice
        s.write(val); // write to signal
        b.write(val); // write to buffer
        wait(1, SC_SEC); // wait after 1 s
      }
      val++; // value change
    }
  }
  void consumer1() {
    while (true) {
      std::cout << sc_time_stamp() << ": consumer1 receives " << s.read() << std::endl;
      wait(); // receives from signal
    }
  }
  void consumer2() {
    while (true) {
      std::cout << sc_time_stamp() << ": consumer2 receives " << b.read() << std::endl;
      wait(); // receives from buffer
    }
  }
};

int sc_main(int, char*[]) {
  BUFFER buffer("buffer");
  sc_start(4, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: consumer1 receives 1
0 s: consumer2 receives 1
1 s: consumer2 receives 1
2 s: consumer1 receives 2
2 s: consumer2 receives 2
3 s: consumer2 receives 2***
# Communication: port
三个关键概念：
1. 接口（Interface）：
    a) 是一个从sc_interface派生的抽象类，但不从sc_object派生。
    b) 包含一组需要在从该接口派生的通道中定义的纯虚函数。
2. 端口（Port）： 
    a) 提供一种方式，使模块能够独立于其实例化上下文进行编写。 
    b) 将接口方法调用转发到绑定到端口的通道。 
    c) 定义了一组由包含端口的模块所需的服务（由端口的类型确定）。
3. 通道（Channel）： 
    a) sc_prim_channel是所有原始通道的基类。 
    b) 通道可以提供可以使用接口方法调用范式调用的公共成员函数。 
    c) 原始通道应实现一个或多个接口。
简而言之： 
* 端口需要服务，接口定义服务，通道实现服务。
* 一个端口可以连接到（绑定）一个通道，如果该通道实现了该端口所需的接口。
* 一个端口是对通道的指针。

何时使用端口：
1. 如果一个模块要调用属于该模块之外的通道的成员函数，则应通过该模块的端口使用接口方法调用进行调用。否则，这被认为是不良的编码风格。
2. 然而，可以直接调用在当前模块内实例化的通道的成员函数。这称为无端口通道访问。
3. 如果一个模块要调用子模块中的通道实例的成员函数，则应通过子模块的导出进行调用。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(MODULE1) { // defines one module
  sc_signal<int> s; // a signal (channel) inside the module
  sc_port<sc_signal_out_if<int> > p; // a port used to write to an outside channel
  SC_CTOR(MODULE1) {
    SC_THREAD(selfWrite); // a process to write to own channel
    SC_THREAD(selfRead); // a process to read from own channel
    sensitive << s; // triggered by value change on the channel
    dont_initialize();
    SC_THREAD(outsideWrite); // a process to write to an outside channel
  }
  void selfWrite() {
    int val = 1; // init value
    while (true) {
      s.write(val++); // write to own channel
      wait(1, SC_SEC); // repeat after 1 s
    }
  }
  void selfRead() {
    while (true) {
      std::cout << sc_time_stamp() << ": reads from own channel, val=" << s.read() << std::endl; // read from own channel
      wait(); // receives from signal
    }
  }
  void outsideWrite() {
    int val = 1; // init value
    while (true) {
      p->write(val++); // write to an outside channel, calls the write method of the outside channel. p is a pointer.
      wait(1, SC_SEC);
    }
  }
};
SC_MODULE(MODULE2) { // a module that reads from an outside channel
  sc_port<sc_signal_in_if<int> > p; // a port used to read from an outside channel
  SC_CTOR(MODULE2) {
    SC_THREAD(outsideRead); // a process to read from an outside channel
    sensitive << p; // triggered by value change on the channel
    dont_initialize();
  }
  void outsideRead() {
    while (true) {
      std::cout << sc_time_stamp() << ": reads from outside channel, val=" << p->read() << std::endl; // use port to read from the channel, like a pointer.
      wait(); // receives from port
    }
  }
};

int sc_main(int, char*[]) {
  MODULE1 module1("module1"); // instantiate module1
  MODULE2 module2("module2"); // instantiate module2
  sc_signal<int> s; // declares a signal (channel) outside module1 and moudle2
  module1.p(s); // binds (connects) port p of module1 to channel (signal) s
  module2.p(s); // binds port p of module2 to channel s
  sc_start(2, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: reads from own channel, val=1
0 s: reads from outside channel, val=1
1 s: reads from own channel, val=2
1 s: reads from outside channel, val=2***
# Communication: export
export：
1. 允许模块向其父模块提供接口。
2. 将接口方法调用转发到与导出绑定的通道。
3. 定义了包含export的模块提供的一组服务。

何时使用export：
1. 通过export提供接口是替代简单地实现接口的一种方式。
2. 显式export允许单个模块实例以结构化的方式提供多个接口。
3. 如果一个模块需要在子模块中调用属于通道实例的成员函数，则应该通过子模块的export进行调用。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(MODULE1) { // defines one module
  sc_export<sc_signal<int>> p; // an export for other modules to connect
  sc_signal<int> s; // a signal (channel) inside the module. If not using export, the channel need to be defined outside module1.
  SC_CTOR(MODULE1) {
    p(s); // bind an export to an internal channel
    SC_THREAD(writer); // a process to write to an internal channel
  }
  void writer() {
    int val = 1; // init value
    while (true) {
      s.write(val++); // write to an internal channel
      wait(1, SC_SEC);
    }
  }
};
SC_MODULE(MODULE2) { // a module that reads from an export
  sc_port<sc_signal_in_if<int>> p; // a port used to read from an export of another module
  SC_CTOR(MODULE2) {
    SC_THREAD(reader); // a process to read from an outside channel
    sensitive << p; // triggered by value change on the channel
    dont_initialize();
  }
  void reader() {
    while (true) {
      std::cout << sc_time_stamp() << ": reads from outside channel, val=" << p->read() << std::endl; // use port to read from the channel, like a pointer.
      wait(); // receives from port
    }
  }
};

int sc_main(int, char*[]) {
  MODULE1 module1("module1"); // instantiate module1
  MODULE2 module2("module2"); // instantiate module2
  module2.p(module1.p); // connect module2's port to module1's export. No need to declare a channel outside module1 and module2.
  sc_start(2, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: reads from outside channel, val=1
1 s: reads from outside channel, val=2***
# Communication: port 2 port
到目前为止，我们讨论了以下情况：
1. 通过通道连接同一模块的两个进程： 
    process1() --> channel --> process2()
2. 通过端口和通道连接不同模块的两个进程： 
    module1::process1() --> module1::port1 --> channel --> module2::port2 --> module2::process2()
3. 通过导出连接不同模块的两个进程：
    module1::process1() --> module1::channel --> module1::export1 --> module2::port2 --> module2::process2()
在这些情况下，都需要使用通道来连接端口。有一个特殊情况允许端口直接连接到子模块的端口。即， module::port1 --> module::submodule::port2
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(SUBMODULE1) { // a submodule that writes to channel
  sc_port<sc_signal_out_if<int>> p;
  SC_CTOR(SUBMODULE1) {
    SC_THREAD(writer);
  }
  void writer() {
    int val = 1; // init value
    while (true) {
      p->write(val++); // write to channel through port
      wait(1, SC_SEC);
    }
  }
};
SC_MODULE(SUBMODULE2) { // a submodule that reads from channel
  sc_port<sc_signal_in_if<int>> p;
  SC_CTOR(SUBMODULE2) {
    SC_THREAD(reader);
    sensitive << p; // triggered by value change on the channel
    dont_initialize();
  }
  void reader() {
    while (true) {
      std::cout << sc_time_stamp() << ": reads from channel, val=" << p->read() << std::endl;
      wait(); // receives from channel through port
    }
  }
};
SC_MODULE(MODULE1) { // top-level module
  sc_port<sc_signal_out_if<int>> p; // port
  SUBMODULE1 sub1; // declares submodule
  SC_CTOR(MODULE1): sub1("sub1") { // instantiate submodule
    sub1.p(p); // bind submodule's port directly to parent's port
  }
};
SC_MODULE(MODULE2) {
  sc_port<sc_signal_in_if<int>> p;
  SUBMODULE2 sub2;
  SC_CTOR(MODULE2): sub2("sub2") {
    sub2.p(p); // bind submodule's port directly to parent's port
  }
};

int sc_main(int, char*[]) {
  MODULE1 module1("module1"); // instantiate module1
  MODULE2 module2("module2"); // instantiate module2
  sc_signal<int> s; // define channel outside module1 and module2
  module1.p(s); // bind module1's port to channel, for writing purpose
  module2.p(s); // bind module2's port to channel, for reading purpose
  sc_start(2, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: reads from channel, val=1
1 s: reads from channel, val=2***
# Communication: specialized ports
除了使用基本的sc_port类来声明端口，还有其他各种专门的端口类可以使用不同的通道类型或提供额外的功能：
1. sc_in: 一个用于信号的专用端口类。
2. sc_fifo_in：用于从fifo读取的专用端口类。
3. sc_fifo_out：用于写入fifo的专用端口类。
4. sc_in<\bool> 和sc_in<sc_dt::sc_logic>: value_changed(), pos(), neg()
5. sc_inout: 用于信号的专用端口类：value_changed(), initialize()
6. sc_inout<\bool>和sc_inout<sc_dt::sc_logic>: 为双值信号提供额外成员函数的专用端口类：value_changed(), initialize(), pos(), neg()
7. sc_out: 从sc_inout派生的类，除了由于是派生类所带来的一些内在差异（例如构造函数和赋值运算符），它与sc_inout完全相同。
8. sc_in_resolved: 用于已解析信号的专用端口类。它与从类`sc_in<sc_dt::sc_logic>`派生的端口行为类似。唯一的区别是，类sc_in_resolved的端口应绑定到类sc_signal_resolved的通道，而类`sc_in<sc_dt::sc_logic>`的端口可以绑定到类`sc_signal<sc_dt::sc_logic,WRITER_POLICY>`或类sc_signal_resolved的通道。
9. sc_inout_resolved: 用于已解析信号的专用端口类。它与从类`sc_inout<sc_dt::sc_logic>`派生的端口行为类似。唯一的区别是，类sc_inout_resolved的端口应绑定到类sc_signal_resolved的通道，而类`sc_inout<sc_dt::sc_logic>`的端口可以绑定到类`sc_signal<sc_dt::sc_logic,WRITER_POLICY>`或类sc_signal_resolved的通道。
10. sc_out_resolved是从sc_inout_resolved派生的类，除了由于是派生类所带来的一些内在差异（例如构造函数和赋值运算符），它与sc_inout_resolved完全相同。
11. sc_in_rv是一个用于已解析信号的专用端口类，它与从类`sc_in<sc_dt::sc_lv<W>>`派生的端口行为类似。唯一的区别是，一个sc_in_rv类端口必须绑定到sc_signal_rv类的通道上，而一个`sc_in<sc_dt::sc_lv<W>>`类的端口可以绑定到`sc_signal<sc_dt::sc_lv<W>,WRITER_POLICY>`或sc_signal_rv类的通道上。
12. sc_inout_rv是一个用于已解析信号的专用端口类，它与从类`sc_inout<sc_dt::sc_lv<W>>`派生的端口行为类似。唯一的区别是，一个sc_inout_rv类端口必须绑定到sc_signal_rv类的通道上，而一个`sc_inout<sc_dt::sc_lv<W>>`类的端口可以绑定到`sc_signal<sc_dt::sc_lv<W>,WRITER_POLICY>`或sc_signal_rv类的通道上。
13. sc_out_rv是从sc_inout_rv派生的类，除了由于是派生类所带来的一些内在差异（例如构造函数和赋值运算符），它与sc_inout_rv完全相同。

一个基本的`sc_port<sc_signal_inout_if<int>>` 只能访问信号通道提供的以下成员函数：
1. read()
2. write()
3. default_event() // 当通过 sc_sensitive 类的 operator<< 定义静态灵敏度时调用。
4. event() // 检查是否发生了事件，返回 true/false
5. value_changed_event() // 值改变事件

一个 `sc_port<sc_signal_inout_if<bool>>` 可以访问signal<\bool>提供的附加的成员函数： 
6. posedge() // 如果值从 false 变为 true，则返回 true 
7. posedge_event() // 值从 false 变为 true 的事件 
8. negedge() // 如果值从 true 变为 false，则返回 true 
9. negedge_event() // 值从 true 变为 false 的事件

一个特殊的 sc_inout<> 的端口提供了以下额外的成员函数： 
10. initialize() // 在端口绑定到通道之前初始化端口的值 
11. value_changed() // 用于在端口绑定到通道之前建立敏感性（指针未初始化）

当底层信号通道的类型为 bool 或 sc_logic 时， sc_inout<\bool> 提供了两个额外的成员函数： 
12. pos() // 在绑定端口之前建立敏感性 
13. neg() // 在绑定端口之前建立敏感性

在上面列出的成员函数中： 
1~9 由信号通道提供，可通过 "port->method()" 访问； 
10~13 由专用端口提供，可通过 "port.method()" 访问。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(WRITER) {
  sc_out<bool> p1, p2; // specialized ports
  SC_CTOR(WRITER) {
    SC_THREAD(writer);
    p1.initialize(true); // #10, initialize default value to true
  }
  void writer() {
    bool v = true;
    while (true) {
      p1->write(v); // #2 write through port
      v = !v; // value change
      wait(1, SC_SEC); // repeat after 1 s
    }
  }
};
SC_MODULE(READER) {
  sc_in<bool> p1, p2; // specialized ports
  SC_CTOR(READER) {
    SC_THREAD(reader1);
    sensitive << p1 << p2; // #3 default_event(), same as p->default_event() or p.default_event()
    dont_initialize();
    SC_THREAD(reader2);
    sensitive << p1.value_changed(); // #11, sensitive to value change event of an un-bound port
    dont_initialize();
    SC_THREAD(reader3);
    sensitive << p1.neg(); // #13, sensitive to neg event of an un-bound port
    dont_initialize();
    SC_THREAD(reader4);
    sensitive << p1.pos(); // #12, sensitive to pos event of an un-bound port
    dont_initialize();
  }
  void reader1() {
    while (true) {
      std::cout << sc_time_stamp() << ": default_event. p1 = " << p1->read() << "; p1 triggered? " << p1->event() << "; p2 triggered? " << p2->event() << std::endl; // #1 read(), #4 event()
      wait();
    }
  }
  void reader2() {
    while (true) {
      std::cout << sc_time_stamp() << ": value_changed_event. p1 = " << p1->read() <<  std::endl; // #1 read()
      wait();
    }
  }
  void reader3() {
    while (true) {
      std::cout << sc_time_stamp() << ": negedge_event. p1 = " << p1->read() << "; negedge = " << p1->negedge() << std::endl; // #8, if negedge happened
      wait();
    }
  }
  void reader4() {
    while (true) {
      std::cout << sc_time_stamp() << ": posedge_event. p1 = " << p1->read() <<  "; posedge = " << p1->posedge() << std::endl; // #6, if posedge happened
      wait();
    }
  }
};

int sc_main(int, char*[]) {
  WRITER writer("writer"); // instantiate writer
  READER reader("reader"); // instantiate reader
  sc_signal<bool> b1, b2; // declare boolean signal channel
  writer.p1(b1); // port binding
  writer.p2(b2);
  reader.p1(b1);
  reader.p2(b2);
  sc_start(4, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: posedge_event. p1 = 1; posedge = 1
0 s: value_changed_event. p1 = 1
0 s: default_event. p1 = 1; p1 triggered? 1; p2 triggered? 0
1 s: negedge_event. p1 = 0; negedge = 1
1 s: value_changed_event. p1 = 0
1 s: default_event. p1 = 0; p1 triggered? 1; p2 triggered? 0
2 s: posedge_event. p1 = 1; posedge = 1
2 s: value_changed_event. p1 = 1
2 s: default_event. p1 = 1; p1 triggered? 1; p2 triggered? 0
3 s: negedge_event. p1 = 0; negedge = 1
3 s: value_changed_event. p1 = 0
3 s: default_event. p1 = 0; p1 triggered? 1; p2 triggered? 0***
# Communication: port array
在声明一个端口时：
1. 第一个参数是接口名称，同时也是端口的类型： 
    一个端口只能绑定到从端口类型派生的通道，或者绑定到另一个端口，或者绑定到一个从端口类型派生的类型的export。
2. 第二个参数是一个可选的整数值，用于指定端口实例可以绑定的最大通道实例数： 
    a) 默认值为1。 
    b) 如果值为0，则该端口可以绑定任意数量的通道实例。 
    c) 将端口绑定到比允许的数量更多的通道实例是错误的。
3. 第三个参数是一个可选的sc_port_policy类型的端口策略，用于确定多端口绑定的规则和未绑定端口的规则： 
    a) \[default] SC_ONE_OR_MORE_BOUND: 该端口应绑定到一个或多个通道，最大数量由第二个参数的值确定。在推理结束时，端口保持未绑定是错误的。 
    b) SC_ZERO_OR_MORE_BOUND: 该端口应绑定到零个或多个通道，最大数量由第二个参数的值确定。在推理结束时，端口保持未绑定是可以的。 
    c) SC_ALL_BOUND: 该端口应绑定到正好等于第二个参数值指定的通道实例数，不多也不少，前提是该值大于零。
        1) 如果第二个参数的值为0，则策略SC_ALL_BOUND与策略SC_ONE_OR_MORE_BOUND具有相同的含义。
		2) 在推理结束时，端口保持未绑定或绑定到比第二个参数所需的通道少是错误。
将给定的端口多次绑定到一个给定的通道是不对的，无论是直接绑定还是通过另一个端口绑定。
另一种定义端口数组的方法是使用C/C++数组语法：sc_port p\[10] 或 vector<sc_port\<IF>> p(10);。

示例：

1. sc_port\<IF>                                                          // 绑定到恰好1个通道实例
2. sc_port<IF,0>                                                       // 绑定到1个或多个通道实例，没有上限
3. sc_port<IF,3>                                                       // 绑定到1、2或3个通道实例
4. sc_port<IF,0,SC_ZERO_OR_MORE_BOUND>       // 绑定到0个或多个通道实例，没有上限
5. sc_port<IF,1,SC_ZERO_OR_MORE_BOUND>       // 绑定到0个或1个通道实例
6. sc_port<IF,3,SC_ZERO_OR_MORE_BOUND>       // 绑定到0个、1个、2个或3个通道实例
7. sc_port<IF,3,SC_ALL_BOUND>                            // 绑定到恰好3个通道实例
8. sc_port<IF, 3>                                                     // 3个端口的数组，每个绑定到恰好1个通道实例
9. vector<sc_port\<IF>> p(3)                                  // 3个端口的数组，每个绑定到恰好1个通道实例
```c++
#include <systemc>
#include <vector> // used to define a vector of ports
using namespace sc_core;

SC_MODULE(WRITER) {
  sc_port<sc_signal_out_if<int>> p1; // #1: exactly 1 channel
  sc_port<sc_signal_out_if<int>, 0> p2; // #2: 1 or more channels, no upper limit
  sc_port<sc_signal_out_if<int>, 3> p3; // #3: 1, 2, or 3 channels
  sc_port<sc_signal_out_if<int>, 0, SC_ZERO_OR_MORE_BOUND> p4; // #4: 0 or more channels, no upper limit
  sc_port<sc_signal_out_if<int>, 1, SC_ZERO_OR_MORE_BOUND> p5; // #5: 0 or 1 channels
  sc_port<sc_signal_out_if<int>, 3, SC_ZERO_OR_MORE_BOUND> p6; // #6: 0, 1, 2, or 3 channels
  sc_port<sc_signal_out_if<int>, 3, SC_ALL_BOUND> p7; // #7: exactly 3 channels
  std::vector<sc_port<sc_signal_out_if<int>>> p9; // #9: vector of port
  SC_CTOR(WRITER) : p9(3) { // init p9 to size of 3
    SC_THREAD(writer);
  }
  void writer() {
    int v = 1;
    while (true) {
      p9[0]->write(v); // write to p9[0]
      p7[1]->write(v++); // write to p7[1]
      wait(1, SC_SEC);
    }
  }
};
SC_MODULE(READER) {
  sc_port<sc_signal_in_if<int>> p1; // #1: exactly 1 channel
  sc_port<sc_signal_in_if<int>, 0> p2; // #2: 1 or more channels, no upper limit
  sc_port<sc_signal_in_if<int>, 3> p3; // #3: 1, 2, or 3 channels
  sc_port<sc_signal_in_if<int>, 0, SC_ZERO_OR_MORE_BOUND> p4; // #4: 0 or more channels, no upper limit
  sc_port<sc_signal_in_if<int>, 1, SC_ZERO_OR_MORE_BOUND> p5; // #5: 0 or 1 channels
  sc_port<sc_signal_in_if<int>, 3, SC_ZERO_OR_MORE_BOUND> p6; // #6: 0, 1, 2, or 3 channels
  sc_port<sc_signal_in_if<int>, 3, SC_ALL_BOUND> p7; // #7: exactly 3 channels
  std::vector<sc_port<sc_signal_in_if<int>>> p9; // #9: exactly 3 channels
  SC_CTOR(READER) : p9(3) { // init p9 to size of 3
    SC_THREAD(reader7);
    sensitive << p7; // sensitive to any element of port array p7
    dont_initialize();
    SC_THREAD(reader9);
    sensitive << p9[0] << p9[1] << p9[2]; // sensitive to any element of port array p9
    dont_initialize();
  }
  void reader7() {
    while (true) {
      std::cout << sc_time_stamp() << "; reader7, port 0/1/2 = " << p7[0]->read() << "/" << p7[1]->read() << "/" << p7[2]->read() << std::endl;
      wait();
    }
  }
  void reader9() {
    while (true) {
      std::cout << sc_time_stamp() << "; reader9, port 0/1/2 = " << p9[0]->read() << "/" << p9[1]->read() << "/" << p9[2]->read() << std::endl;
      wait();
    }
  }
};

int sc_main(int, char*[]) {
  WRITER writer("writer"); // instantiate writer
  READER reader("reader"); // instantiate reader
  // declare channels
  sc_signal<int> s1; // 1 channel
  std::vector<sc_signal<int>> s2(10); // 10 channels
  std::vector<sc_signal<int>> s3(2); // 2 channel
  // leave s4 un-bound
  sc_signal<int> s5; // 1 channel
  std::vector<sc_signal<int>> s6(2); // 2 channels
  std::vector<sc_signal<int>> s7(3); // 3 channels
  // #8 is same as #9, omitted
  std::vector<sc_signal<int>> s9(3); // 3 channels
  // bind ports
  writer.p1(s1); // #1
  reader.p1(s1); // #1
  for (unsigned int i = 0; i < s2.size(); ++i) { // #2
    writer.p2(s2[i]);
    reader.p2(s2[i]);
  }
  for (unsigned int i = 0; i < s3.size(); ++i) { // #3
    writer.p3(s3[i]);
    reader.p3(s3[i]);
  }
  // s4 un-bound
  writer.p5(s5); // #5
  reader.p5(s5); // #5
  for (unsigned int i = 0; i < s6.size(); ++i) { // #6
    writer.p6(s6[i]);
    reader.p6(s6[i]);
  }
  for (unsigned int i = 0; i < s7.size(); ++i) { // #7
    writer.p7(s7[i]);
    reader.p7(s7[i]);
  }
  for (unsigned int i = 0; i < s9.size(); ++i) { // #9
    writer.p9[i](s9[i]);
    reader.p9[i](s9[i]);
  }
  sc_start(2, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s; reader9, port 0/1/2 = 1/0/0
0 s; reader7, port 0/1/2 = 0/1/0
1 s; reader9, port 0/1/2 = 2/0/0
1 s; reader7, port 0/1/2 = 0/2/0***
# Primitive Channel
sc_prim_channel：
1. 是所有原始通道的基类。
2. 为原始通道提供独特的访问调度器更新阶段的途径。
3. 不包含层次结构、端口或仿真过程。
4. 与分层通道相同的是，原始通道可以提供公共成员函数，可以使用接口方法调用范式进行调用。
提供如下成员函数：
a) request_update(): 
    调度程序将通道的更新请求排队 
b) async_request_update(): 
    1. 调度程序以线程安全的方式将通道的更新请求排队。能够可靠地从systemC内核之外的操作系统线程调用。 
    2. 不建议从在systemC内核上下文中执行的函数中调用 
c) update(): 
    1. 调度程序在更新阶段响应对request_update或async_request_update的调用时回调。 
    2. 应用程序可以覆盖此成员函数。sc_prim_channel本身对此函数的定义什么都不做。 
    3. 通常只读取和修改当前对象的数据成员并创建delta通知。 
    4. 不应该： 
        a) 如果在当前对象的基类中覆盖，则除update函数之外，不要调用类sc_prim_channel的任何成员函数。 
        b) 不带参数调用类sc_event的成员函数notify()以创建立即通知。 
        c) 调用类sc_process_handle的任何成员函数进行进程控制（例如挂起或杀死）。
        d) 更改除当前对象的数据成员之外的任何存储的状态。 
        e) 读取除当前对象之外的任何原始通道实例的状态。 
        f) 调用其他通道实例的接口方法。特别是，update成员函数不应写入任何信号。 
d) next_trigger() 
e) wait()
一个通道需要实现一个或多个接口，因此需要继承自接口类（sc_interface的基类）。接口提供了通道所需的方法。
```c++
#include <systemc>
#include <string>
using namespace sc_core;

class GENERATOR_IF : public sc_interface { // interface for interrupt generator
public:
  virtual void notify() = 0;
};
class RECEIVER_IF : public sc_interface { // interface for interrupt receiver
public:
  virtual const sc_event& default_event() const = 0; // needed for sensitive
};
class INTERRUPT : public sc_prim_channel, public GENERATOR_IF, public RECEIVER_IF { // interrupt class
public:
  INTERRUPT(sc_module_name name) : sc_prim_channel(name) {} // constructor, construct sc_prim_channel
  void notify() { // implement GENERATOR_IF
    e.notify();
  }
  const sc_event& default_event() const { // implement RECEIVER_IF
    return e;
  }
private:
  sc_event e; // private event for synchronization
};
SC_MODULE(GENERATOR) { // interrupt generator class
  sc_port<GENERATOR_IF> p; // port to generate interrupt
  SC_CTOR(GENERATOR) { // constructor
    SC_THREAD(gen_interrupt);
  }
  void gen_interrupt() {
    while (true) {
      p->notify(); // calls notify function of the INTERRUPT channel
      wait(1, SC_SEC);
    }
  }
};
SC_MODULE(RECEIVER) { // interrupt receiver class
  sc_port<RECEIVER_IF> p; // port to receive interrupt
  SC_CTOR(RECEIVER) { // constructor
    SC_THREAD(rcv_interrupt);
    sensitive << p; // monitors interrupt on port p
    dont_initialize();
  }
  void rcv_interrupt() { // triggered upon interrupt
    while (true) {
      std::cout << sc_time_stamp() << ": interrupt received" << std::endl;
      wait();
    }
  }
};

int sc_main(int, char*[]) {
  GENERATOR generator("generator"); // instantiate generator
  RECEIVER receiver("receiver"); // instantiate receiver
  INTERRUPT interrupt("interrupt"); // instantiate interrupt
  generator.p(interrupt); // port binding
  receiver.p(interrupt); // port binding
  sc_start(2, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: interrupt received
1 s: interrupt received***
# Hierarchical Channel
分层通道：
1. 应继承sc_channel基类，该基类与sc_module相同。因此，分层通道是systemC模块。
2. 应从接口继承，以便与端口连接。
像常规的systemC模块一样，分层通道也可能有仿真过程、端口等。
这个例子展示了一个自定义的分层通道，它实现了sc_signal_inout_if\<int>。根据sc_signal_inout_if的定义，我们必须实现以下函数：
1. void write(const int&)
2. const int& read() const
3. const sc_event& value_changed_event() const
4. const sc_event& default_event() const
5. const int& get_data_ref() const
6. bool event() const
```c++
#include <systemc>
using namespace sc_core;

// this is a simple implementation as compared to sc_signal, just to illustrate the concept of a hieracical channel
class SIGNAL : public sc_channel, public sc_signal_inout_if<int> { // declares SIGNAL channel, inherits from sc_chanel and signal_inout_if<int>
public:
  SC_HAS_PROCESS(SIGNAL);
  SIGNAL(sc_module_name name = sc_gen_unique_name("SIG")) : sc_channel(name) {} // constructor, construct base class
  void write(const int& v) { // implements write method
    if (v != m_val) { // update only if value is new
      m_val = v; // update value
      e.notify(); // trigger event
    }
  }
  const int& read() const {
    return m_val;
  }
  const sc_event& value_changed_event() const {
    return e; // return reference to the event
  }
  const sc_event& default_event() const {
    return value_changed_event(); // allows used in static sensitivity list
  }
  const int& get_data_ref() const {
    return m_val;
  }
  bool event() const {
    return true; // dummy implementation, always return true
  }
private:
  int m_val = 0;
  sc_event e;
};

SC_MODULE(TEST) { // a test class
  SIGNAL s; // declares SIGNAL channel
  SC_CTOR(TEST) { // no name provided to s, use default
    SC_THREAD(writer); // register a writer process
    SC_THREAD(reader); // register a reader process
    sensitive << s; // use SIGNAL channel in static sensitivity list
    dont_initialize();
  }
  void writer() {
    int v = 1;
    while (true) {
      s.write(v++); // write to channel
      wait(1, SC_SEC);
    }
  }
  void reader() {
    while (true) {
      std::cout << sc_time_stamp() << ": val = " << s.read() << std::endl; // read from channel
      wait();
    }
  }
};
int sc_main(int, char*[]) {
  TEST test("test"); // instantiate generator
  sc_start(2, SC_SEC);
  return 0;
}
```
上面代码输出结果:
***0 s: val = 1
1 s: val = 2***
# Trace File
一个跟踪文件：
1. 记录仿真过程中按时间顺序排列的值序列。
2. 使用VCD（值变化转储）文件格式。
3. 只能通过sc_create_vcd_trace_file创建和打开。
4. 可以在推理过程中或在仿真过程中的任何时候打开。
5. 包含只能通过sc_trace追踪的值。
6. 在将值记录到该文件之前，应打开跟踪文件，如果自打开文件以来一个或多个delta周期已过，则不应将值记录到给定的跟踪文件。
7. 应由sc_close_vcd_trace_file关闭。在仿真的最后一个delta周期之前，跟踪文件不应被关闭。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(MODULE) { // a module write to a channel
  sc_port<sc_signal<int>> p; // a port
  SC_CTOR(MODULE) {
    SC_THREAD(writer); // a writer process
  }
  void writer() {
    int v = 1;
    while (true) {
      p->write(v++); // write to channel via port
      wait(1, SC_SEC); // write every 1 s
    }
  }
};
int sc_main(int, char*[]) {
  MODULE module("module"); // instantiate module
  sc_signal<int> s; // declares signal channel
  module.p(s); // bind port to channel

  sc_trace_file* file = sc_create_vcd_trace_file("trace"); // open trace file
  sc_trace(file, s, "signal"); // trace "s" under the name of "signal"
  sc_start(5, SC_SEC); // run simulation for 5 s
  sc_close_vcd_trace_file(file); // close trace file
  return 0;
}
```
上面代码输出结果:
***Info: (I702) default timescale unit used for tracing: 1 ps (trace.vcd)***
# Error and Message Report
sc_report:
    1. 表示由函数 `sc_report_handler::report` 生成的报告的实例。
    2. 如果为给定的严重性级别和消息类型设置了 `SC_CACHE_REPORT`，则应用程序可以访问该报告。
    3. 当报告处理程序抛出时，应用程序可以捕获该报告。
sc_report_handler:
    提供用于写入异常情况发生时的文本报告的功能，以及用于定义生成这些报告时要执行的应用程序特定行为的功能。
sc_severity 表示报告的严重性级别：
    1. `enum sc_severity {SC_INFO = 0, SC_WARNING, SC_ERROR, SC_FATAL, SC_MAX_SEVERITY}`;
    2. 有四个严重性级别。`SC_MAX_SEVERITY` 不是严重性级别。将 `SC_MAX_SEVERITY` 传递给需要 `sc_severity` 类型参数的函数是错误的。

`sc_verbosity`提供了可以作为参数传递给类`sc_report_handler`的成员函数`set_verbosity_level`和`report`的指示性详细程度值:
`enum sc_verbosity {SC_NONE = 0, SC_LOW = 100, SC_MEDIUM = 200, SC_HIGH = 300, SC_FULL = 400, SC_DEBUG = 500};`

sc_actions代表一个字，其中字中的每一位都代表一个不同的操作。如果设置了多个位，则应执行所有相应的操作：
1. enum {
     SC_UNSPECIFIED  = 0x0000, //is not an action, serves as the default value meaning that no action has been set.
     SC_DO_NOTHING   = 0x0001, // is a specified action
     SC_THROW        = 0x0002,
     SC_LOG          = 0x0004,
     SC_DISPLAY      = 0x0008,
     SC_CACHE_REPORT = 0x0010,
     SC_INTERRUPT    = 0x0020,
     SC_STOP         = 0x0040,
     SC_ABORT        = 0x0080
    }
2. 每个严重级别都对应一个默认的行为，可以用函数set_action()覆盖默认行为
3. 默认行为：
    a) `#define SC_DEFAULT_INFO_ACTIONS ( SC_LOG | SC_DISPLAY )`
    b) `#define SC_DEFAULT_WARNING_ACTIONS ( SC_LOG | SC_DISPLAY )`
    c) `#define SC_DEFAULT_ERROR_ACTIONS ( SC_LOG | SC_CACHE_REPORT | SC_THROW )`
    d) `#define SC_DEFAULT_FATAL_ACTIONS ( SC_LOG | SC_DISPLAY | SC_CACHE_REPORT | SC_ABORT )`

`void report(sc_severity, const char* msg_type, const char* msg, [int verbosity], const char* file, int line`)生成一个报告并采取对应的操作：
1. 使用作为第一个参数传递的严重程度和作为第二个参数传递的消息类型来确定由于之前对函数 set_actions、stop_after、suppress 和 force 的调用而要执行的action set。
2. 使用所有五个参数值创建 sc_report 类的对象，并将该对象传递给通过成员函数 set_handler 设置的hander实例。
3. 除非设置了动作 SC_CACHE_REPORT，否则不会在调用成员函数 report 后持久保留。在这种情况下，可以通过调用函数 get_cached_reports 来检索该对象。
4. 负责确定要执行的行动集。通过函数 set_handler 设置的处理程序函数负责执行这些行动。
5. 维护报告的计数。无论是否执行或抑制行动，这些计数都应增加，除非由于报告的详细程度级别而忽略报告，在这种情况下，计数不应增加。

set_actions()：
1. 设置成员函数 report 在使用给定的严重性级别、消息类型或两者一起调用时采取的action。
2. 替换之前针对给定的严重性、消息类型或严重性-消息类型对调用的action。

stop_after(): 
	report 应当在为给定的严重性级别、消息类型或严重性消息类型对调用 stop_after 函数的参数 limit 所给定的报告数量时调用 sc_stop。

get_count(): 
    返回由成员函数 report 维护的每个严重性级别、每个消息类型和每个严重性消息类型对所产生的报告的数量。

Verbosity level：
1. int set_verbosity_level(int)：将最大详细程度级别设置为作为参数传递的值，并返回最大详细程度级别的上一个值。
2. int get_verbosity_level()：返回最大详细程度级别的值。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(MODULE) { // a test module
  sc_port<sc_signal<int>> p; // a port
  SC_CTOR(MODULE) { // constructor
    SC_REPORT_WARNING("ctor", "register function"); // gen report to "ctor"
    SC_THREAD(writer); // a writer process
    SC_THREAD(reader); // a reader process
    sensitive << p; // sensitive to p
    dont_initialize();
  }
  void writer() {
    int v = 1;
    while (true) {
      SC_REPORT_INFO("writer", ("write " + std::to_string(v)).c_str()); // gen report to "writer"
      p->write(v++); // write to channel via port
      wait(1, SC_SEC); // write every 1 s
    }
  }
  void reader() {
    while (true) {
      SC_REPORT_INFO("reader", ("read " + std::to_string(p->read())).c_str()); // gen report to "reader"
      wait();
    }
  }
};
int sc_main(int, char*[]) {
  sc_report_handler::set_log_file_name("report.log"); // initialize report
  sc_report_handler::set_actions("writer", SC_INFO, SC_LOG); // INFO of "writer" saved in log, no display

  MODULE module("module"); // instantiate module
  sc_signal<int> s; // declares signal channel
  module.p(s); // bind port to channel

  SC_REPORT_INFO("main", "simulation starts"); // gen report to "main"
  sc_start(2, SC_SEC); // run simulation for 2 s
  SC_REPORT_INFO("main", "simulation ends"); // gen report to "main"
  return 0;
}
```
上面代码输出结果:
***Warning: ctor: register function
In file: /home/jiang/code/c_example/example.cpp:7
Info: main: simulation starts
Info: reader: read 1
Info: reader: read 2
Info: main: simulation ends***
# Customized Data Type
`sc_signal<T>`和`sc_fifo<T>`可以与各种数据类型一起使用。SystemC已经支持了内置数据类型当作T。
为了在sc_signal和sc_fifo中使用自定义数据类型，需要为数据类型实现以下成员函数： 
1. 赋值运算符，即operator=()：用于读写方法 
2. 相等运算符，即operator\==()：由sc_signal用于value_changed_event() 
3. 输出流，即ostream& operator<<()：用于打印数据结构 
4. sc_trace()：允许数据类型与systemC跟踪工具一起使用；允许使用waveform查看器查看跟踪的数据。
```c++
#include <systemc>
#include <ostream>
using namespace sc_core;

struct CUSTOMIZED_TYPE {
  int x, y; // member variables
  CUSTOMIZED_TYPE(int x = 0, int y = 0) : x(x), y(y) {} // constructor
  CUSTOMIZED_TYPE& operator=(const CUSTOMIZED_TYPE& rhs) { // assignment operator, needed for read() write()
    x = rhs.x;
    y = rhs.y;
    return *this;
  }
  bool operator==(const CUSTOMIZED_TYPE& rhs) { // equality operator, needed for value_changed_event()
    return x == rhs.x && y == rhs.y;
  }
};
std::ostream& operator<<(std::ostream& os, const CUSTOMIZED_TYPE& val) { // streaming output, needed for printing
  os << "x = " << val.x << "; y = " << val.y << std::endl;
  return os;
}
inline void sc_trace(sc_trace_file*& f, const CUSTOMIZED_TYPE& val, std::string name) { // needed for tracing
  sc_trace(f, val.x, name + ".x");
  sc_trace(f, val.y, name + ".y");
}

SC_MODULE(MODULE) { // test module
  sc_signal<CUSTOMIZED_TYPE> s; // customized signal
  SC_CTOR(MODULE) { // constructor
    SC_THREAD(writer); // writer process
    SC_THREAD(reader); // reader process
    sensitive << s; // sensitive to customized signal s
    dont_initialize();
  }
  void writer() {
    int x = 1; // init signal
    int y = 2;
    while (true) {
      s.write(CUSTOMIZED_TYPE{x++, y++}); // write to signal
      wait(1, SC_SEC); // wait 1 s
    }
  }
  void reader() {
    while (true) {
      std::cout << sc_time_stamp() << ": receives " << s.read() << std::endl; // read from signal
      wait(); // wait for value_changed_event
    }
  }
};

int sc_main(int, char*[]) {
  MODULE module("module"); // instantiate module
  sc_trace_file* file = sc_create_vcd_trace_file("trace"); // open trace file
  sc_trace(file, module.s, "customized_type"); // trace customized signal
  sc_start(2, SC_SEC); // run simulation for 2 s
  sc_close_vcd_trace_file(file); // close trace file
  return 0;
}
```
上面代码输出结果:
***0 s: receives x = 1; y = 2
Info: (I702) default timescale unit used for tracing: 1 ps (trace.vcd)
1 s: receives x = 2; y = 3***
# Clock
sc_clock 是一个从 sc_signal 类派生而来的预定义原始通道，用于模拟数字时钟信号的行为。 通过接口 sc_signal_in_if 可以访问与时钟相关的值和事件。

构造函数： 
```c++
sc_clock(
  constchar*name_, // unique module name
  double period_v_, // the time interval between two consecutive transitions from false to true, also equal to the time interval between two consecutive transitions from true to false. Greater than zero, default is 1 nanosecond.
  sc_time_unit period_tu_, // time unit, used for period
  double duty_cycle_, // the proportion of the period during which the clock has the value true. Between 0.0 and 1.0, exclusive. Default is 0.5.
  double start_time_v_, // the absolute time of the first transition of the value of the clock (false to true or true to false). Default is zero.
  sc_time_unit start_time_tu_,
  bool posedge_first_ = true ); // if true, the clock is initialized to false, and changes to true at the start time. Vice versa. Default is true.
```
example
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(CLOCK) {
  sc_port<sc_signal_in_if<bool>> clk; // a port to access clock
  SC_CTOR(CLOCK) {
    SC_THREAD(thread); // register a thread process
    sensitive << clk; // sensitive to clock
    dont_initialize();
  }
  void thread() {
    while (true) {
      std::cout << sc_time_stamp() << ", value = " << clk->read() << std::endl; // print current clock value
      wait(); // wait for next clock value change
    }
  }
};

int sc_main(int, char*[]) {
  sc_clock clk("clk", 10, SC_SEC, 0.2, 10, SC_SEC, false); // 10s period, 2s true, 8s false, start at 10s, start at false.
  CLOCK clock("clock"); // instantiate module
  clock.clk(clk); // bind port
  sc_start(31, SC_SEC); // run simulation for 31 s
  return 0;
}
```
上面代码输出结果:
***10 s, value = 0
18 s, value = 1
20 s, value = 0
28 s, value = 1
30 s, value = 0***
# Process: CTHREAD
SC_CTHREAD的描述如下：
1. 在SystemC 2.0中已被弃用。但在第二个参数是事件查找器的情况下仍然受支持。
2. 在注册一个进程时需要时钟。
3. 没有像SC_METHOD或SC_THREAD似的单独的灵敏度列表。
4. 每当指定的时钟边沿发生时就会被激活。
```c++
#include <systemc>
using namespace sc_core;

SC_MODULE(MODULE) {
  sc_in<bool> clk; // need event_finder method, cannot use basic sc_port
  SC_CTOR(MODULE) {
    SC_CTHREAD(cthread1, clk); // sensitive to clk pos edge
    // no static sensitivity, thus, cannot use dont_initialize()
    SC_CTHREAD(cthread2, clk.pos()); // sensitive to clk pos edge
    SC_CTHREAD(cthread3, clk.neg()); // sensitive to clk neg edge
  }
  void cthread1() {
    while (true) {
      wait(); // wait for clk pos edge; wait() right after while loop to avoid initialization
      std::cout << sc_time_stamp() << ", cthread1, value = " << clk->read() << std::endl;
    }
  }
  void cthread2() {
    while (true) {
      wait(); // wait for clk pos edge
      std::cout << sc_time_stamp() << ", cthread2, value = " << clk->read() << std::endl;
    }
  }
  void cthread3() {
    while (true) {
      wait(); // wait for clk neg edge
      std::cout << sc_time_stamp() << ", cthread3, value = " << clk->read() << std::endl;
    }
  }
};

int sc_main(int, char*[]) {
  sc_clock clk("clk", 10, SC_SEC, 0.2, 10, SC_SEC, false); // 10s period, 2s true, 8s false, start at 10s, start at false.
  MODULE module("module"); // instantiate module
  module.clk(clk); // bind port
  sc_start(31, SC_SEC); // run simulation for 31 s
  return 0;
}
```
上面代码输出结果:
***20 s, cthread3, value = 0
28 s, cthread2, value = 1
28 s, cthread1, value = 1
30 s, cthread3, value = 0***