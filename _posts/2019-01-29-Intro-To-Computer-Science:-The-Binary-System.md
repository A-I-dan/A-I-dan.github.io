---
layout:     post
title:      "Introduction to Computer Science: The Binary System"
subtitle:   "Computer Science: Binary"
date:       2019-01-24 12:00:00
author:     "A.I. Dan"
---
<img src='https://github.com/A-I-dan/blog/blob/master/images/binary-1695478_1920.jpg?raw=true'>

# The Binary System

[Bi-nary:](https://www.etymonline.com/word/binary)
>From Late Latin binarius "consisting of two," from bini "twofold, two apiece, two-by-two" (used especially of matched things), from bis "double".


<b><i>There are 10 types of people in the world. Those who understand binary, and those who don't.</i></b>

In this post I will be going over the <b>Binary System</b>. This is a topic usually covered in the first introductory class to computer science and is something that is fun to know as well.

<hr>

## What Is The Binary Number System?

In an introductory course to computer science, you will hear about <b>binary sequences</b>. A binary sequence is a series of 1s and 0s that represent <b>all data</b> in a computer. Yes... all data. This includes your texts, instagram videos and pictures, apps, etc.

The <b>binary number system</b> is using 1s and 0s to represent numbers in our decimal system. The decimal system is the standard number system consisting of 0-9, whereas the binary system uses just 0-1.

Here is a quick little example of binary numbers translating over to the decimal system:

| Binary System | Decimal    System |
| ------------- |:-----------------:|
| 0001          |         1         |
| 0010          |         2         |
| 0011          |         3         |
| 0100          |         4         |
| ...           |        ...        |
| 101010        |        42         |
| 110111        |        55         |

### Why Do Computers Use The Binary System?

For computers, the binary system is very convenient. Compared to the decimal system, hex, or octal systems, binary is the optimal system for computers. 0s and 1s can simply represent <b>off</b> and <b>on</b>. It would be difficult for computers to run on the decimal system, where 0 is "off", 1 is "kind of off", 2 is "barely on", and 8 is "almost fully on". It is not hard to see why the binary system works best when representing switches in the computer. One day we may have <b>quantum computers</b>, which would be able to have these "switches" in multiple states. So maybe with quantum computing, we can use the decimal system for our machines. But until then, we have binary.

<hr>

## Convert Binary To Decimal

Imagine the decimal system. Lets pick a random number - 198.

| 198 In     | The  Decimal |   System |
| ---------- |:------------:| --------:|
| 100        |     + 90     |      + 8 |
| 100s place |  10s place   | 1s place |

In the decimal system, the place values are determined by the powers of 10. It is a <b>base-10</b> system. For example:

![equation](http://mathurl.com/render.cgi?10%5E0%20-%2010%5E1%20-%2010%5E2%20-%2010%5E3%2C%20etc%0A%0A1s%20%20-%2010s%20-%20100s%20-%201000s%2C%20etc%5Cnocache)

The binary system is similar, except the binary system is a <b>base-2</b> system. This means values are determined by the <b>power of 2</b>. For example:

![equation](http://mathurl.com/render.cgi?2%5E0%20-%202%5E1%20-%202%5E2%20-%202%5E3%20-%202%5E4%20-%202%5E5%0A%0A1s%20%20-%202s%20-%204s%20-%208s%20-%2016s%20-%2032s%5Cnocache)

Here is how to determine what a binary number translates to in the decimal system:

(1) Write the binary number and leave some space between each digit.

1  1  1  0

(2) Above each binary digit (starting from the right), write the corresponding power of 2.

8  4  2  1

1  1  1  0

(3) Add up each value with a 1 underneath of it. In this case, the binary number 1110 is equal to the decimal number 14. Because:

![equation](http://mathurl.com/render.cgi?2%5E3%2C%202%5E2%2C%202%5E1%2C%202%5E0%0A%0A1%20-%201%20-%201%20-%200%0A%0A8+%204+2+%200%20%20%3D%2014%0A%5Cnocache)

## Convert Decimal To Binary

We will convert the number 21 to binary. Heres how:

(1) First, write out the powers of 2 and the corresponding digits for the binary system.

![equation](http://mathurl.com/render.cgi?%5Ctextmode%202%5E4%2C%202%5E3%2C%202%5E2%2C%202%5E1%2C%202%5E0%0A%0A16s%2C8s%2C4s%2C2s%2C1s%5Cnocache)

(2) Now, pick out the largest place value that can go into our number - 21. In this example it would be the 16s place. So add a 1 underneath the 16s place value.

![equation](http://mathurl.com/render.cgi?%5Ctextmode%202%5E4%2C%202%5E3%2C%202%5E2%2C%202%5E1%2C%202%5E0%0A%0A16s%2C8s%2C4s%2C2s%2C1s%0A%0A1%5Cnocache)

(3) Subtract 16 from 21. This will get you an answer of 5. Now repeat <b>step (2)</b>, but this time using the number 5. So the largest place value that goes into 5 would be 4. Put a 1 underneath of the 4s place value.

![equation](http://mathurl.com/render.cgi?%5Ctextmode%202%5E4%2C%202%5E3%2C%202%5E2%2C%202%5E1%2C%202%5E0%0A%0A16s%2C8s%2C4s%2C2s%2C1s%0A%0A1%20-%20-%20-%201%20-%20-%20-%20-%5Cnocache)

(4) Now again, subtract 4 (the largest place value to go into 5) from 5 to get 1. Now repeat <b>step (2)</b> using 1. The largest place value to go into 1 is... 1. So mark a 1 underneath the 1s place value.

![equation](http://mathurl.com/render.cgi?%5Ctextmode%202%5E4%2C%202%5E3%2C%202%5E2%2C%202%5E1%2C%202%5E0%0A%0A16s%2C8s%2C4s%2C2s%2C1s%0A%0A1%20-%20-%20-%201%20-%20-%20-1%5Cnocache)

(5) Now fill in all the remaining place values with 0s underneath of them. Add up every number with a 1 below it, this will give you your answer.

![equation](http://mathurl.com/render.cgi?%5Ctextmode%202%5E4%2C%202%5E3%2C%202%5E2%2C%202%5E1%2C%202%5E0%0A%0A16s%2C8s%2C4s%2C2s%2C1s%0A%0A1%20-%200%20-%201%20-0%20-1%5Cnocache)

Now we have our answer. 21 in binary is 10101.

## Conclusion

This was just a short introduction to the binary system. I plan on writing more Introduction to Computer Science-like posts soon to cover all of the basics. 
