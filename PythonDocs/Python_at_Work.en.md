Working with Python
===================

Introduction
------------

This introduction to programming in Python is for those who are not familiar with the language, or do not use it often, and need a handy reference guide to follow the code examples for simulations in the book. A *quick guide* to quickly empower even those who do not know how to program to do computer experiments. It is for those who prefer to learn by doing, who want to see the results of their code immediately, and who want to quickly get their bearings without getting lost in implementation details or lengthy explanations of theoretical computer science. It is not intended to be a comprehensive introduction nor even a formal course in scientific computing, but a hybrid of a **superintensive course - crash course** and a **summary sheet - cheat sheet**, suitable for those in a hurry to "get their hands dirty" with concrete examples and real projects, the best approach to learning.

As a supplement to these notes for those unfamiliar with programming in Python we recommend, among many excellent texts for getting started, the free online versions of:

- "PY4E Python for everyone: Exploring data using python 3",
    <https://www.py4e.com/html3/>

- "Think Python: How to think like computer scientists,"
    <https://allendowney.github.io/ThinkPython/>

Other very useful resources, online and free, are:

- "Python Tutorial", <https://docs.python.org/3/tutorial/>

- "Python CheatSheet", <https://www.pythoncheatsheet.org/>

- "The Hitchhiker′s Guide to Python", <https://docs.python-guide.org/>

In the first part we will review how to get started in Python: you can choose between working online, installing an integrated software suite for scientific computing on your personal computer, or traditional installation of the language, the advanced interactive interpreter, the integrated development environment, and a Web-based development, execution, and sharing environment. In the second part a quick introduction to the basics of the language: variables, control statements, functions, modules, importing libraries,...

Working Tools
-------------------

### Installation

To get started with Python there are three main alternatives:

1.  Install nothing on your computer and immediately start working and collaborating online on the Internet, with python and Jupyter Notebook, using with any browser a service such as

    SageMath **CoCalc** - <https://cocalc.com/>
    or Google **Colab** - <https://colab.google/>.

2.  Download and install from the official site **Anaconda 3**:

    <https://anaconda.com/download>

    a software distribution for data science and artificial intelligence, freely available for Linux, Windows and Mac, which includes a complete installation of the Python language and major add-on modules. You can follow the official installation guide

    <https://www.anaconda.com/docs/getting-started/anaconda/install>

3.  Install in language on your computer, and use an interactive interpreter com IPython, or an integrated development environment such as Spyder. Python is available for virtually every operating system: Linux, Windows, macOS. In Linux distributions a version of the language is usually already installed. The official documentation for installation on Linux or Windows or macOS:

    Python Setup and Usage - <https://docs.python.org/3/using/>

The easiest choice, and the best for collaborating in a team, is of course the first (CoCalc or Colab), for working locally on one's own computer the highly recommended choice is the second (install Anaconda), the third way is the most traditional but the least advisable.

### Jupyter Notebook: Computational Web Environment.

Jupyter Notebook is an interactive Web-based environment, i.e., usable simply with a browser, for programming in Python and other languages, analyzing data, writing documentation, and creating scientific reports. It allows you to write code, visualize results, add text, mathematical formulas and graphs, all in a single, easily shared document (notebook). Key features include:

- Creating interactive documents that combine code, output and documentation

- Support for visualization of data and graphs

- Ability to export results in a variety of formats

- Easy sharing of analysis and results

Jupyter Notebook is a powerful and flexible tool for writing, executing and documenting Python code interactively, ideal for both beginners and advanced users.

**Project Jupyter** is a nonprofit organization created to support data science and scientific computing in all programming languages through the development of open source software, open standards, and services for interactive computing. The name of the Jupyter project is a reference to the three main programming languages supported by Jupyter, which are Julia, Python and R, and also an homage to Galileo's notebooks (notebooks) on the discovery of Jupiter's moons:

<https://jupyter.org/>

The term notebook (notebook, notepad) is used, depending on the context, for the Jupyter web application, the Jupyter Python web server, or the Jupyter document format, and is a computational interface style introduced by MathCad in 1986, and became popular with software such as Mathematica, Maple, and SageMath.

A Jupyter Notebook *document*, in a file with the extension **.ipynb**, is a JSON document that contains an ordered list of input/output cells, cell structure, metadata, code in any supported language, text (in Markdown format), mathematics, graphs, and *rich output*, i.e., multimedia data(audio, video, images, ..., with ASCII Base64 encoding).

A Jupyter Notebook document can be converted to numerous high formats: HTML, LaTeX, PDF, ReStructuredText, Markdown, Python.

A *Jupyter kernel* is a program responsible for handling various types of requests (e.g., code execution) and providing a response. Kernels communicate with other Jupyter components using a distributed messaging platform. Kernels are available to allow programming in about 50 different languages. For more information:

<https://docs.jupyter.org/en/stable/>

### CoCalc: SageMath cloud computing service.

CoCalc (Collaborative Calculation) is an online platform offering a cloud environment for real-time scientific computing, scheduling and collaboration, based on **Jupyther Notebooks**. Most useful for teaching and research, it eliminates installation hassles and offers advanced tools for collaboration, project management and scientific writing.

### Colab: Google cloud computing service

Google Colab (Google Colaboratory) is a free service from Google that allows you to write and run Python code directly from your browser, without having to install anything on your computer. Also like CoCalc, Colab is based on the **Jupyter Notebooks**, and allows you to use Python in an easy, collaborative and free way, with advanced hardware resources and all major libraries ready to use.

### Anaconda: Scientific Python Distribution.

Anaconda is an open-source distribution of Python and R specialized for scientific computing, machine learning and data science Provides a collection of over 1,000 preinstalled software packages for working in science.

Key features of Anaconda include:

- Simplified management of packages and virtual environments via conda

- Pre-configured deployment with major scientific libraries

- Support for creating isolated environments for specific projects

- Graphical interface for managing packages and environments

- Ability to use Jupyter Notebook, Spyder IDE and IPython locally

**Anaconda 3** - <https://anaconda.org>,

### Spyder: Integrated Development Environment

Spyder is an open-source integrated development environment (IDE) designed specifically for scientific programming in Python. Written in Python for Python, it was designed by and for scientists, engineers, and data analysts...

Distinctive features of Spyder include:

- Integration with Numpy, SciPy, Pandas, IPython, QtConsole, MatPlotLib, SymPy and other scientific libraries

- Advanced editor with debugging and profiling capabilities

- Built-in interactive console

- Variable explorer for data inspection

- Code quality control tools (Pyflakes, Pylint, Rope)

**Spyder IDE** - <https://www.spyder-ide.org/>

Another useful development environment for beginners in python is IDLE (Integrated Development and Learning Environment), distributed with the language, or for more experienced programmers the community version of PyCharm.

### IPython: Advanced Interactive Interpreter.

**IPython** (Interactive Python) represents an advanced interactive development environment that extends the functionality of the standard Python interpreter. It offers functionality for interactive parallel computing and can be used from a terminal by simply typing `ipython` instead of `python`. .

IPython supports advanced interactive mode with features such as `name?` to display help and `!command` to execute shell commands.

<https://ipython.org/>

Essential Web Services
----------------------

- **Official Documentation**, mainly in English, with a Tutorial, a Reference Guide, a manual for the Standard Library, and a Setup and Usage guide, partial translation into Italian.

    Documentation: [docs.python.org](https://docs.python.org/3/) 1.

    2.[Tutorial](https://docs.python.org/3/tutorial/index.html)

    3.[Reference Guide](https://docs.python.org/3/reference/)

    4.[Standard library](https://docs.python.org/3/library/)

    5.[Setup and Usage](https://docs.python.org/3/using/).

    6. English translation(partial): [python.it/doc](https://www.python.it/doc/)

- **GitHub** The global reference point for hosting, collaborating and discovering open source Python projects. Here you find both the official Python repository and thousands of libraries, frameworks and tools of all kinds. Essential for versioning, issue tracking, pull requests and community.

- **PyPI (Python Package Index)** The central repository of all open source Python packages. From here you can install (via pip) virtually any public Python library. Each package has its own page with documentation, changelog, and links to the source repo.

- **Stack Overflow** The largest technical Q&A community, most useful for resolving doubts, finding code snippets and best practices.

- **Read the Docs** Hosts documentation for many Python libraries, often updated and searchable online.

- **Awesome Python** A curated collection of the best Python libraries, frameworks, tools and resources, organized by category and updated regularly.

- **SageMath** is a powerful open source system for computational mathematics, built on Python and integrated with many scientific libraries (such as NumPy, SciPy, matplotlib, SymPy, Maxima, GAP, R, etc.).

    It aspires to be an open and free alternative to expensive commercial products such as Mathematica, Maple and Matlab. It can be used online without installing anything with the CoCalc service.

Python in Brief
---------------

***introduction to the language***

- Python was designed for readability and has some similarities to the English language, also influenced by the language of mathematics. and the data types are static.

- In Python we use a carriage return at the end of each command, as opposed to other programming languages that often use semicolons or parentheses.

- In Python there is always a distinction between lower case and upper case.

- In Python you always count starting at 0 (not 1), each index starts at 0.

- Indentation required: Python uses indentation to define blocks of code (4 spaces recommended). Indentation, i.e., indenting lines of program text with whitespace or tabs, is essential in Python. Indentation is used to distinguish blocks of instructions, to be executed together in program flow control, e.g., in defining functions, conditional instructions, instruction loops to be repeated, error handling. Other programming languages often use curly brackets for this purpose.

- Comments: Use '#' for single-line comments.

- In Python you do not have to declare variables at the beginning of the program, and the data type is not fixed. Therefore it is easier for beginners to learn. Other languages such as Perl, Julia, PHP, and JavaScript behave similarly. In contrast, in languages such as C, C++, Java, variables must be declared before using them.

- Meaningful names: Assign descriptive names to variables.

- When a Python program (script) is executed, it is first precompiled into *bytecode*, an intermediate code between source code and machine language, then the bytecode is saved for later reuse, finally interpreted and converted to machine language for actual execution.

- In some cases Python functions are compiled in real time into machine language (Numba platform) or converted to C and compiled as in that language (Cython platform). This greatly speeds up program execution.

- Python also admits an interactive mode, ideal for beginners, in which you enter commands and immediately see the result. In this it is similar to Lisp and software such as SageMath, Octave, MatLab, Mathematica, but different from other programming languages.

-

Variables
---------

A variable can be named in any way as long as it obeys the following rules:

1.  It can be a single word, not a phrase.

2.  Can use only letters, numbers and the underscore character _ , the underscore .

3.  Cannot begin with a number.

4.  Must be a name other than the reserved names of functions, constants and instructions in the language.

**Value assignment** The assignment operator is `=`

Examples:

- `a = b` assigns the value b to the variable a.

- `x = 1` assigns the value 1 to the variable x.

- `spam = 'Hello'` assigns the value 'Hello' to the variable `spam`.

Never confuse the value assignment operator to a variable `=` with the equality operator between two terms of an expression `==`.

**Multiple assignment**

The multiple assignment trick is a shortcut that allows you to assign values to multiple variables in the same line of code.

Examples:

**single assignment**

``` {.python language="Python"}
size = 'fat'
    color = 'orange'
    disposition = 'strong'
```

**multiple assignment**

``` {.python language="Python"}
size, color, disposition = ['fat', 'orange', 'strong']
```

Multiple assignment is also useful for swapping values between two variables:

Example:

``` {.python language="Python"}
a, b = 'Alice', 'Bob'
a, b = b, a
print(a)
print(b)
```

Standard data types
---------------------

A program handles many different types of data, such as numbers, characters, text. The most common are:

| Name   | Description  |
|------------|--------------------------------------------------------|
|  int |  Integer number, signed                                      |
| float | Decimal number in floating point                            |
| complex|  complex number, such as 2 + 3j ( j is the imaginary unit) |
|  bool | Logical value (boolean),   *true* or *false*                |
|  str  | String, a sequence of characters, such as a word or phrase  | 


Notes:

1.  A period is always used for decimal numbers. 

- Decimal numbers are written either with a period followed by the decimal digits or in exponential (scientific) notation, with `e` to indicate the power of 10 by which to multiply

- For complex numbers we use **j** as the imaginary unit (the square of j equals -1, $j^2 = -1$ ) like engineers, instead of **i** which is used instead in mathematics.

- String characters are stored in Unicode (character encoding method independent of the language and computer system used)

Examples:

- `-2, -1, 0, 1, 2, 3, 4, 5` are integers ( **int**).

- `-1.25, -1.0, --0.5, 0.0, 0.5, 1.0, 1.25` are floating-point numbers ( **float**).

- `1.2e3`, which is equivalent to `1200`, is always a floating-point number in base-10 exponential notation.

- `'a', `aaa', `Hello!', `11 cats', `345'` are strings (**str**), and are always enclosed in superscripts (quoted).

A number of data types are defined internally for compound data structures, formed with the preceding simple (scalar) types

|  Name Description|
|------- ----------------------------------------------------------------------|
|  set  | Set, an unordered, modifiable collection of any data|
|  list  | List, an ordered collection of data, with an index|
|  tuple |  Tuple, an unmodifiable list|
|  dict |  dictionary, a table consisting of key-value pairs|

Other useful data types are not built into the language, but provided by additional modules.

| Name  | Module | Description|
|----------|----------- --------------------------------------------------------|
| fraction | fractions  | fraction, rational number such as 2/3, 0.1|
| decimal  | decimal | decimal with assigned precision, alternative to float|
| ndarray  | numpy  | vector, matrix, multidimensional grid of data|
| datetime | datetime | a combination of date, time, and time zone|

Some basic functions
--------------------

**most common functions** Many functions are defined in the base language, here are some of the most common ones

 | Function   | Description                          |
 |------------|---------------------------------------|
 |print()     | prints a result string as output|
 |input()     | reads in input data entered by the user|
 |help()      | help information about a function or form|
 |type()      | checks the data type of a variable|
 |abs(n)      | Returns the absolute value of a number|
 |round(x)    | Rounds a number|
 |divmod(m,n) | Returns the quotient and remainder of a division|
 |len(string) | Returns the length of a string|

**The print function**

Prints the string or variables passed in parentheses as an argument

``` {.python language="Python"}
print('Hello World!')
```

``` {.python language="Python"}
a = 42
print('Hello World! The answer is ', a)
```

To modify the print we use the keywords end: terminator and sep: separator

``` {.python language="Python"}
print('Hello', end='')
print('World')
```

``` {.python language="Python"}
print('cats', 'dogs', 'mice')
```

``` {.python language="Python"}
print('cats', 'dogs', 'mice', sep=',')
```

**The input function**

The input function is used to read the value of a piece of data, which must be entered by the program executor

Example:

``` {.python language="Python"}
print('What is your name?') # asks for the name
myName = input()
print('A real pleasure to meet you,', myName)
```

**The help function**

About the functions

``` {.python language="Python"}
help(print)
```

Without arguments is an interactive tool for consulting the documentation of the language

``` {.python language="Python"}
help()
```

**The type function**

If you are not sure of the data type you use the special **type** function to display it:

    * type(27) results in *int*
    * type('hello') results in *str*

**The conversion functions str, int, float, ...**

 | Function | Description |
 |------------| ---------------------|
 |int() | Converts to an integer|
 |float() | Converts to a decimal number |
 |complex() | Converts to a complex number |
 |bool() | Returns boolean value (true or false) |
 |chr() | Converts a Unicode code to a character |
 |str() | Converts to a string |
 |bin() | Converts a number to binary |
 |oct() | Converts a number to octal |
 |hex() | Converts a number to hexadecimal |
 |dict() | Converts data to a dictionary |
 |tuple() | Converts data to a tuple |
 |set() | Converts data to a set |
 |frozenset() | Converts data to a frozenset |
 |bytearray() | Converts to an editable list of bytes |
 |bytes() | Converts to an immutable list of bytes |


Data type names also correspond to functions to convert one type to another. For example: `int('345')` converts the string '345' to the integer 345,  `float(27)`  converts the integer 27 to the decimal number 27.0.

Integer to String or Float:

``` {.python language="Python"}
str(29)
```

``` {.python language="Python"}
print('I am {} years old.'.format(str(29)))
```

``` {.python language="Python"}
str(-3.14)
```

Float to Integer:

``` {.python language="Python"}
int(7.7)
```

``` {.python language="Python"}
int(7.7) + 1
```

Comments
--------

To make the code more understandable, it is a good practice to add many comments.

Comment on a line:

``` {.python language="Python"}
# ``This is an inline comment.
```

Multiline comment:

``` {.python language="Python"}
# This instead is a comment
# on multiple lines
```

Code with a comment:

``` {.python language="Python"}
a = 1 # initialization
```

Note the two spaces before the comment.

**The value None**

``` {.python language="Python"}
spam = print('Hello!')
spam is None
```

Note: Never compare "None" with the "==" operator. Always use "is".

Operators
---------

**Mathematical operators**

Mathematical operators in order of precedence:

 |Operator | Operation | Example |
 |-------|---------|-----------------|
 |`**` | Exponent, power elevation | `2 ** 3 = 8`.|
 |`%` | Modulus/Rest of integer division | `22 % 8 = 6`.|
 |`//` | Division between integers | `22 // 8 = 2`|
 |`/` | Division between integers or decimals | `22 / 8 = 2.75`|
 |`*` | Multiplication | `3 * 3 = 9`.|
 |`-` | Subtraction | `5 - 2 = 3`|
 |`+` | Addition | `2 + 2 = 4`|

Mathematical operators with assignment

In addition to performing a mathematical operation they assign the result to a variable, e.g. ` k += 1 ` adds 1 to the `k` counter.

 |Operator | Operation | Example |
 |---------|-----------|-------------- |
 |`+=` | Addition and assignment  | x  += 1 adds 1 to x |
 |`-=` | Subtraction and assignment  | x  -= 1 subtracts 1 to x |
 |`*=` | Multiplication and assignment  | x  *= 2 multiplies x by 2 |
 |`/=` | Division and assignment  | x /= 3 divides x by 3 |
 |
Expressions such as in mathematics are used to alter precedence.

Example of expressions in the interactive interpreter:

``` {.python language="Python"}
2 + 3 * 6
20
```

``` {.python language="Python"}
(2 + 3) * 6
30
```

``` {.python language="Python"}
2 ** 8
256
```

``` {.python language="Python"}
23 / 7
3.2857142857142856
```

``` {.python language="Python"}
23 // 7
3
```

``` {.python language="Python"}
23 % 7
2
```

``` {.python language="Python"}
(5 - 1) * ((7 + 1) / (3 - 1))
16.0
```

**Logical Operators (Boolean)**

|  Operator |  Description                                                         |
|-----------|----------------------------------------------------------------------|
|  and | (expr1) and (expr2) , true if both expressions are true                   |
|  or | (expr1) or (expr2), true if either is true                                 |
|  not | not (expr), true if the expression is false                               |
|  is  (expr) | is True (False), expression is true (or false)                     |
|  is  not (expr) | is not True (False), the expression is not true (or not false) |


**Comparison operators**

|  Operator |  Meaning|
|----------- --------------------|
|  `==` | Equal|
|  `!=` | Different|
|  `<` | Less than|
|  `>` | Greater than|
|  `<=` | Less than or equal|
|  `>=` | Greater than or equal to|

An expression with these operators is true or false depending on whether the condition is met or not.

Flow Control
-------------------

**The if statement**

``` {.python language="Python"}
if name == 'Alice':
    print('Hi, Alice.')
```

**The else statement**

``` {.python language="Python"}
name = 'Bob'

if name == 'Alice':
    print('Hi, Alice.')
else:
    print('Hello, stranger.')
```

**The elif statement**

``` {.python language="Python"}
name = 'bob'
age = 5

if name == 'Alice':
    print('Hi, Alice.')
elif age < 12:
    print('You are not Alice, kiddo.')
```

``` {.python language="Python"}
name = 'bob'
age = 30

if name == 'Alice':
    print('Hi, Alice.')
elif age < 12:
    print('You are not Alice, kiddo.')
else:
    print('You are neither Alice nor a little kid.')
```

Cycles
-----

**Conditional cycles**

While followed by a condition is used to define a block of instructions that is repeated until the condition is met

``` {.python language="Python"}
spam = 0

while spam < 5:
    print('Hello, world.')
    spam = spam + 1
```

**The break statement**

If a break statement is reached during program execution, it immediately exits the while loop

``` {.python language="Python"}
while True:
    print('Please type your name.')
    name = input()
    if name == 'your name':
        break

print('Thank you!')
```

**The continue instruction**

If a contina instruction is reached during program execution, it returns to the beginning of the loop.

``` {.python language="Python"}
while True:
    print('Who are you?')
    name = input()
    if name != 'Joe':
        continue
    print('Hello, Joe. What is the password? (It is a fish.)')
    password = input()
    if password == 'swordfish':
        break

print('Access granted.')
```

**The range()** function.

|  Function  |  Description|
|--------------|-------------------------------------------------------|
|  range(n) | Creates a sequence of integers, from 0 to n - 1|
|  range(i,j,k) |  Creates a sequence from i to j (excluded) with increment k|

The resulting list, of type range, is a sequence of integers.

The function `range ()`  can also be called with three arguments. The first two arguments will be the start and end values, and the third will be the step. The step, or increment, is the amount by which the variable is incremented after each iteration. A negative increment is defined for a countdown.

Examples:

``` {.python language="Python"}
range(20)
range(1, 20, 2):
range(10, 0, -1):
```

**Iterative cycles**

The for statement is used to define the repetition of a block of statements for a range of values of a variable, a finite loop.

Examples:

``` {.python language="Python"}
print('My name is')
for i in range(5):
    print('Jimmy Five Times ({})'.format(str(i)))
```

``` {.python language="Python"}
for i in range(0, 10, 2):
   print(i)
```

``` {.python language="Python"}
for i in range(5, -1, -1):
    print(i)
```

If you do not need, within a loop, the value of the iteration variable, you use `_`,anonymous variable, in the iteration for statement, for example:

``` {.python language="Python"}
print('My name is')
for i in range(10):
    print('Jimmy Ten Times')
```

Definition of Functions
-----------------------

**The def instruction**

The def instruction is used to define new functions

``` {.python language="Python"}
def hello(name):
    print('Hello {}'.format(name))
```

**The return statement and the returned values**.

When you create a function using the def statement, you can specify the returned value with a return statement. A return statement consists of the key *return* followed by the value or expression that the function should return.

``` {.python language="Python"}
import random
def getAnswer(answerNumber):
    if answerNumber == 1:
        return 'It is certain'
    elif answerNumber == 2:
        return 'It is decidedly so'
    elif answerNumber == 3:
        return 'Yes'
    elif answerNumber == 4:
        return 'Reply hazy try again'
    elif answerNumber == 5:
        return 'Ask again later'
    elif answerNumber == 6:
        return 'Concentrate and ask again'
    elif answerNumber == 7:
        return 'My reply is no'
    elif answerNumber == 8:
        return 'Outlook not so good'
    elif answerNumber == 9:
        return 'Very doubtful'

r = random.randint(1, 9)
fortune = getAnswer(r)
print(fortune)
```

Error Handling
---------------

**exceptions, try statements ..., except**

Usually when an error is made during execution, the program stops. To change this behavior you use a block of instructions

``` {.python language="Python"}
try:
    <some operation that may give an error>
except <Error type> as e:
    <do something else instead of exiting the program>
```

Example:

``` {.python language="Python"}
def spam(divideBy):
    try:
        return 42 / divideBy
    except ZeroDivisionError as e:
        print('Error: Invalid argument: {}'.format(e))

print(spam(2))
print(spam(12))
print(spam(0))
print(spam(1))
```

Packages and add-ons
-----------------------------

Importing modules
----------------------

In Python, a *module* or library is called a collection of functions, data and constants (classes, objects and methods in the language of object-oriented programming). A module is typically contained in a computer file with the extension .py.

To use a module, you use the import statement at the beginning of a program

``` {.python language="Python"}
import math
```

here the standard math function library is imported, from the file math.py. In the rest of the program you can use functions, constants and module data types using math as a prefix. Example:

``` {.python language="Python"}
import math
math.pi
math.cos( 0.15 * math.pi )
```

You can also change the name of the module with the import command, and use the new prefix in the program

``` {.python language="Python"}
import math as m
m.cos( 0.15 * m.pi )
```

Finally, you can import some functions and constants from the module into the main program namespace and use them directly without prefixing them

``` {.python language="Python"}
from math import pi,sin
sin(0.3 * pi)
```

It is also possible to import all the names of a module directly into the program. But it is a very bad practice, generating errors and confusion, which makes sense to use only as a beginner, when taking the first steps in the world of programming and the python language, and writing very simple programs

``` {.python language="Python"}
from math import *
cos(0.2 * pi)
```

``` {.python language="Python"}
import random

for i in range(5):
    print(random.randint(1, 10))
```

``` {.python language="Python"}
import random, sys, os, math
```

``` {.python language="Python"}
from random import *.
```

**Information about a module**

    import math
    help(math)
    help(math.cosh)

**The dir function**

An internal function of the language, dir(), lists all the names of functions and variables in a module

``` {.python language="Python"}
import math
dir(math)
```

If you call dir() without arguments, all known names in the program are listed

``` {.python language="Python"}
dir()
```

Standard library (built-in)
----------------------------

**Standard Python library**: Some of the most important modules distributed with the language

- **os** Interface with the operating system: file management, directories, environment variables.

- **os.path** Management of paths and file names in a portable way.

- **sys** Python interpreter parameters and functions, handling input/output, path, program output.

- **argparse** Parsing of arguments from the command line.

- **math** Basic mathematical functions (trigonometry, logarithms, constants such as pi and e).

- **cmath** Mathematical functions for complex numbers: trigonometric, logarithmic, exponential, polar/rectangular conversions.

- **random** Generation of random numbers, random choice from sequences.

- **datetime** Handling of dates and times, time calculations.

- **time** Functions to measure time, delays, timestamps.

- **re** Regular expressions for searching and manipulating strings.

- **json** Reading and writing data in JSON format.

- **pickle** Serialization and deserialization of Python objects.

- **urllib** Accessing and manipulating URLs, HTTP requests.

- **csv** Reading and writing CSV files.

- **zipfile, tarfile** Reading and writing compressed ZIP and TAR archives.

- **sqlite3** relational database management on SQLite files.

Python libraries
---------------

**Science and Data Science Libraries**

- **mpmath** Library for arbitrary-precision floating-point arithmetic for real and complex numbers. Supports advanced calculations with configurable precision (10 digits or 1000 digits), numerical integration, differentiation, root finding, and linear algebra.

- **NumPy**: Numerical computation, multidimensional arrays.

- **Pandas**: Tabular data analysis and manipulation.

- **Matplotlib/Seaborn/Plotly**: Data visualization, static and interactive graphs.

- **Pillow**: Image manipulation.

- **OpenCV**: Computer vision and processing of images and video in real time.

- **SciPy**: Advanced scientific computing.

**Neural networks and machine learning**.

- **PyTorch** Environment for machine learning and neural networks with GPU support, suitable for advanced research

- **FastAI** Interface for rapidly developing neural networks, built with PyTorch, useful in both teaching and research

- **TensorFlow** Google library to build and train artificial intelligence models, for advanced research and production.

- **Keras** Simple interface to TensorFlow, for rapidly creating deep neural networks, useful for teaching and experimentation.

- **JAX** Google library for high-performance scientific computing, with application to neural networks.

- **Scikit-learn** Tools for traditional machine learning (classification, regression, clustering).

**Web and Database Development**

- **Requests**: simple and powerful HTTP client.

- **SQLAlchemy**: ORM and toolkit for relational Databases.

- **Flask**: Micro-framework for web apps and APIs, lightweight and flexible.

- **Django**: Complete framework for robust and scalable web applications.

- **FastAPI**: Modern framework for high-performance APIs.

Development of new Modules
------------------------

A **module** in Python is simply a file that contains Python code: functions, variables, classes, etc. It is used to organize code into reusable parts.

- If you write functions in a file called `my_module.py`, that is a module.

- You can use code from that file in other Python files by writing `import my_module`.

**Example:**

``` {.python language="Python"}
# file: hello.py
def hello():
    print("Hello!")
```

``` {.python language="Python"}
# file: main.py
import greetings

greetings.hello() # print: hello!
```

To create a file in Python that contains functions that can be reused by other scripts, follow these simple steps:

**Create a Python file with your functions**.

For example, call the file `my_module.py`:

``` {.python language="Python"}
# my_module.py

def greet(name):
    return f "Hello, {name}!"

def sum(a, b):
    return a + b
```

**Save the file in the same folder as the script that will use it**

Make sure that `mio_module.py` is in the same directory as the main script, or in a directory included in the PYTHONPATH.

**Import the functions into your main script**.

In the file you want to use the functions (e.g. `main.py`):

``` {.python language="Python"}
# main.py

import my_module

print(my_module.greet("Alice"))    # Output: Hello, Alice!
print(my_module.sum(3, 5)) # Output: 8
```

Or, to import only some functions:

``` {.python language="Python"}
from my_module import greeting

print(greet("Bob"))                 # Output: Hello, Bob!
```

- You can create forms with any name (except those reserved by Python).

- You can organize many functions in one or more files and import them as needed.

A module.py file can contain both function definitions and the main script to be executed. In Python, main is the special name given to code that is executed directly as the main program. If a script is executed directly, the special variable `name` within that file takes on the value `main`.

This feature is typically exploited with the condition:

``` {.python language="Python"}
if __name__ == "__main__":
    ....
```

the code that follows this condition is executed directly as the main script


The Zen of Python
----------------

From [The Zen of Python](https://www.python.org/dev/peps/pep-0020/), by Tim Peters:

    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one--and preferably only one--obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!


**Easter Egg**

``` {.python language="Python"}
import this
```

References
-----------

**Books**

- Charles Severance (Dr.Chuck), "Python for Everybody: Exploring data with python 3," 2018.

- Allen B. Downey, "Think Python: Think like a computer scientist" , 3rd ed. 2025.

- Al Sweigart, "Automate the Boring Stuff with Python: Practical Programming for Total Beginners" , 3rd ed. 2025,

- Kenneth Reitz, Tanya Schlusser,"The Hitchhiker′s Guide to Python: Best Practices for Development," 2016.

- David Beazley, Brian K. Jones,"The Python Cookbook, 3rd ed., (2013).

- Peter Farrell, "Math Adventures with Python: An Illustrated Guide to Exploring Math with Code," 2019

- Amit Saha, "Doing Math with Python: Using Programming to Explore Algebra, Statistics, Calculus, and More! ", 2020

- Mark Newman, "Computational Physics", ch. 2 "Python Programming for Physicists",2012

- Rubin H. Landau,Manuel J. Páez,Cristian C. Bordeianu, "Computational Physics: Problem Solving with Python", 3rd ed. (2024)

**Web Resources**

- "Python Tutorial", <https://docs.python.org/3/tutorial/>

- "Python CheatSheet", <https://www.pythoncheatsheet.org/>

- "The Hitchhiker′s Guide to Python", <https://docs.python-guide.org/>

- Python Docs: <https://docs.python.org>

- "Python for Everybody", <https://www.py4e.com/>,

- "Think Python 3", <https://greenteapress.com/wp/think-python-3rd-edition/>,

- "Automate the Boring Stuff with Python", <https://automatetheboringstuff.com/>
