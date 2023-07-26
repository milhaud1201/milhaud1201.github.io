---
title: Flutter를 위한 Dart 예외처리문
date: 2023-07-22T09:23:28.272Z
categories: [Flutter, Dart]
tags: [flutter, dart]		# TAG는 반드시 소문자
---

```dart
void main() {
  ///예외처리 : 프로그램이 진행 중일 때, 의도하였거나 / 의도하지 않은 상황에 대해서
  ///프로그램적으로 오류가 발생했을 때, 대처하는 방법
  ///try - catch 문 : 가장 기본적인 예외처리문 / 많이 쓰이는 예외처리 문이기도 함.
  ///on 문
  ///throw / rethrow 문
  ///

  int num1 = 10;
  try {
    ///예외가 발생할 것으로 예상되는 코드
    print(10 ~/ 0);
  } catch(error, stack) {
    print(error);
    print(stack);
  } finally {
    ///예외가 발생 했던, 하지 않았던, try문 / catch문 이후에 실행하고자 하는 코드
    print("예외처리 문을 지나쳤습니다.");
  }

  print("위의 에러 때문에 동작을 하지 않습니다.");


  ///try on  exception 예외처리
  ///

  int? num;

  try {
    ///예외가 발생할 것으로 예상되는 코드
    print(num!);
  } on UnsupportedError catch(e, s) {
    print("~/ 해당 연산자는 0으로 나눌 수 없습니다.");
  } on TypeError catch (e, s) {
    print("Null 값이 참조 되었습니다.");
  } catch (e, s) {
    print("모르는 에러가 발생했습니다.");
  }

  print("위의 에러 때문에 동작을 하지 않습니다.");


  ///throw / rethrow문
  ///throw는 예외를 만들어서 던지는 문이다.
  ///rethrow는 예외에서 벗어나도록 던지는 문이다.
  ///

  try {
    ///예외가 발생할 것으로 예상되는 코드
    throw Exception("Unknown Error");
  } on UnsupportedError catch(e, s) {
    print("~/ 해당 연산자는 0으로 나눌 수 없습니다.");
  } on TypeError catch (e, s) {
    print("Null 값이 참조 되었습니다.");
  } catch (e, s) {
    rethrow;
  }

  print("위의 에러 때문에 동작을 하지 않습니다.");
}
```
