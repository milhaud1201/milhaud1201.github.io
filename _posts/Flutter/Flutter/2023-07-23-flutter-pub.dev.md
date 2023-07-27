---
title: Flutter 기본 SDK와 오픈소스 Widget
date: 2023-07-23T09:23:28.272Z
categories: [Flutter, Flutter]
tags: [flutter]		# TAG는 반드시 소문자
---

### pub.dev
* https://pub.dev/
![Alt text](img/image.png)

### keyword
* 기본 SDK
* 오픈소스 Widget
* Widget Tree
* Stateless Widget
* Stateful Widget



## 기본 SDK의 MaterialApp, Scaffold를 활용한 Hello, Flutter 만들기
<p align="center">
    <img src="img/Simulator Screenshot - iPhone 14 Pro Max - 2023-07-27 at 12.09.36.png" alt="drawing" width="200"/>
</p>


```dart
import 'package:flutter/material.dart';

void main() {
  runApp(
    MaterialApp(
      home: Scaffold(
        body: TestWidget()
      ),
    ),
  );
}

class TestWidget extends StatelessWidget {
  const TestWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Center(
        child: Text(
          "Hello, Flutter",
          style: TextStyle(
              fontSize: 60,
              color: Colors.black
          ),
        ),
      ),
    );
  }
}

```