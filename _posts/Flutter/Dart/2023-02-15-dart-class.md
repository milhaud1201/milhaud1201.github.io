---
title: Flutter를 위한 Dart class 문법
date: 2023-02-15T09:23:28.272Z
categories: [Flutter, Dart]
tags: [flutter, dart]		# TAG는 반드시 소문자
---

빠르게 서비스를 구현하기 위해서 어떤 프레임워크를 사용할까 서칭하다가 Flutter를 공부하고 있는데 HTML, CSS, Javascript가 필요 없이 오로지 Dart 언어를 사용하기 때문에 Dart를 꼭 알아야 한다.

해당 포스터는 dart.dev에서 Language tour docs와 노마드코더의 Dart 강의를 참고하였다.

# 1. Classes
Dart에서 작성하는 거의 모든 코드는 클래스에 포함된다. 클래스는 `object` 즉, 만들 수 있는 개체를 말한다. 개체 자체는 특정 데이터와 논리는 가지고 있다.

## 1-1. Constructors

class와 동일한 이름을 가진 함수를 생성하여 constructor를 선언한다. 이때 선택적으로, **Named constructors**에 설명된 additionla identifier를 추가할 수 있다.


dart에서 생성자(constructor) 함수는 클래스 이름과 같아야 한다.

```dart
class Player {
	// 변수에 값을 할당할 때는 late를 씀.
	late final String name;
	late final int age;

	// constructor 함수는 클래스 이름과 같아야 함.
	Player(String name){
	this.name = name;
	}
}

void main(){
// Player 클래스의 인스턴스 생성
var player = Player("milhaud", 20);
```

위의 생성자 함수는 다음과 같이 줄일 수 있다.

```dart
class Player {
	// 변수에 값을 할당할 때는 late를 씀.
	final String name;
	final int age;

	// Constructor, with syntactic sugar for assignment to members.
	Player(this.name, this.age);
	}
}
```


## 1-2. Named constructors

[Named constructor](https://dart.dev/language/constructors#named-constructors)를 사용하여 클래스에 대한 여러 constructors를 구현하거나 추가적인 명확성을 제공한다.

```dart
const double xOrigin = 0;
const double yOrigin = 0;

class Point {
  final double x;
  final double y;

  Point(this.x, this.y);

  // Named constructor
  Point.origin()
      : x = xOrigin,
        y = yOrigin;
}
```

### API로 데이터를 받을 때 클래스로 구현

json format 같은 데이터를 받으면 Flutter Dart class로 바꿔야 한다. 
API로부터 여러 Player가 들어있는 목록을 받는다고 가정하였다.

```dart
class Player {
  final String name;
  int xp;
  String team;

  /// name constructor property 초기화
  Player.fromJson(Map<String, dynamic> playerJSON)
      : name = playerJSON['name'],
        xp = playerJSON['xp'],
        team = playerJSON['team'];

  void sayHello() {
    print("Hi my name is $name");
  }
}

void main() {
  var apiData = [
    {"name": "dokyeong", "team": "red", "xp": 0, "age": 30},
  ];

  apiData.forEach((playerJson) {
    var player = Player.fromJson(playerJson);
    player.sayHello();
  });
}
```

	Hi my name is dokyeong

## 1-3 Enumerated types

[Enumerated types](https://dart.dev/language/enum)은 enumerations이나 enums라고 부른다. Enum는 고정된 상수 값을 나타내는데 사용되는 특별한 종류의 클래스이다. 모든 enums는 자동으로 Enum class를 확장한다. 따라서, subclassed, implemented, mixed in, 명시적으로 인스턴스화 할 수 없다.

### enums 사용하기

simple enumerated type을 선언하려면 enum 키워드를 사용하고 열거하려는 값을 나열한다.

```dart
enum Color { red, green, blue }
```

다른 정적 변수 static variable과 마찬가지로 나열된 값에 접근한다.

```dart
final favoriteColor = Color.blue;
if (favoriteColor == Color.blue) {
  print('Your favorite color is blue!');
}
```

## 1-4 Abstract Clasees

추상화 클래스는 다른 클래스들이 직접 구현해야하는 특정 필드와 메소드를 모아놓은 클래스이다. 인스턴스화를 할 수 없는 추상화 클래스를 정의하기 위해 **abstract** modifier를 이용한다. 메소드의 이름과 반환 타입과 상속받는 모든 클래스가 가지고 있어야 하는 메소드를 정의하고 있다. 

```dart
// This class is declared abstract and thus
// can't be instantiated.
abstract class AbstractContainer {
  // Define constructors, fields, methods...

  void updateChildren(); // Abstract method.
}
```

class는 extends를 사용하여 서브클래스를 생성하여 확장할 수 있다. 

extends로 클래스를 확장하고 super를 사용하여 superclass를 참조한다:

```dart
abstract class Television {
  void turnOn() {
    _illuminateDisplay();
    _activateIrSensor();
  }
  // ···
}

class SmartTelevision extends Television {
  void turnOn() {
    super.turnOn();
    _bootNetworkInterface();
    _initializeMemory();
    _upgradeApps();
  }
  // ···
}
```

## 1-5 Inheritance

Dart에는 단일 상속 [single inheritance](https://dart.dev/language)이 있다. 여기서 super 키워드는 부모 클래스의 name 필드에 직접적으로 접근하는 방식이다.

```dart
class Orbiter extends Spacecraft {
  double altitude;

  Orbiter(super.name, DateTime super.launchDate, this.altitude);
}
```

선택적으로 `@override` 주석으로 부모 클래스의 객체를 받아올 수 있다.

다른 예시:

```dart
class Human {
  final String name;
  Human({required this.name}); // constructor class
  void sayHello() {
    print("Hi my name is $name");
  }
}

enum Team { red, blue }

class Player extends Human {
  final Team team;

  Player({
    required this.team,
    required String name,
  }) : super(name: name); // super

  @override
  void sayHello() {
    super.sayHello(); //Human class의 sayHello 메소드
    print('and I play for ${team}');
  }
}

void main() {
  var player = Player(
    team: Team.red,
    name: 'dokyeong',
  );
  player.sayHello();
}
```

## 1-6 Mixins

Mixin은 생성자가 없는 클래스를 의미한다. 여러 클래스 계층 구조에서 코드를 재사용하는 방법이다.

```dart
mixin Piloted {
  int astronauts = 1;

  void describeCrew() {
    print('Number of astronauts: $astronauts');
  }
}
```

mixin의 기능을 클래스에 추가하려면 with 키워드를 사용한다. 여기서 mixin은 하나 이상 사용할 수 있다.

```dart
class Musician extends Performer with Musical {
  // ···
}

class Maestro extends Person with Musical, Aggressive, Demented {
  Maestro(String maestroName) {
    name = maestroName;
    canConduct = true;
  }
}
```

extends는 확장한 그 클래스는 부모 클래스가 되고, 자식 클래스는 super를 통해 부모 클래스에 접근할 수 있다. mixin의 with은 단순히 mixin 내부의 속성과 메소드를 가져오는 것이다.

# Reference
https://dart.dev/language
https://nomadcoders.co/dart-for-beginners  
https://brunch.co.kr/brunchbook/dartforflutter