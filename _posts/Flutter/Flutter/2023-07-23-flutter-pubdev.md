---
title: Flutter 기본 SDK와 오픈소스 Widget
date: 2023-07-23T09:23:28.272Z
categories: [Flutter, Flutter]
tags: [flutter]		# TAG는 반드시 소문자
---

### pub.dev
* https://pub.dev/
![Alt text](/assets/img/to/pub_dev_home.png)

### keyword
* 기본 SDK
* 오픈소스 Widget
* Widget Tree
* Stateless Widget
* Stateful Widget



## 기본 SDK의 MaterialApp, Scaffold를 활용한 Hello, Flutter 만들기

![image](/assets/img/to/flutter_material_widget.png){: width="200" height="200"}


```dart
import 'package:flutter/material.dart';

void main() {
  runApp(
    MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          actions: [
            IconButton(onPressed: () {
              print("Tab!");
            }, icon: Icon(Icons.home)),
            Icon(Icons.play_arrow)
          ],
          centerTitle: false,
          title: Text("This is appbar"),
        ),
        body: TestWidget(),
        floatingActionButton: FloatingActionButton(
          child: Icon(Icons.bug_report),
          onPressed: () {
            print("Tab! FAB!");
          },
        ),
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

## Keynote
Appbar 위젯에서 title 텍스트를 입력하고 conterTitle을 false로 지정하면 기본 가운데 정렬을 오른쪽 정렬로 바꿔준다.  