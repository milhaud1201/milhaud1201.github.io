---
title: Algorithms - 기본적인 정렬 (버블/선택/삽입/셀)
date: 2023-03-03T09:23:28.272Z
categories: [Algorithms, Array]
tags: [algorithms, array]		# TAG는 반드시 소문자
---

# 정렬 알고리즘
정렬(sort)은 여러 데이터로 구성된 리스트에서 값의 크기 순서에 따하 데이터를 재배치하는 것이다.

* 기본적인 정렬 알고리즘
    * 버블 정렬, **선택 정렬**, **삽입 정렬**, 셸 정렬
* 개선된 성능을 갖는 정렬 알고리즘
    * 합병 정렬, **퀵 정렬**, 힙 정렬




# 1. 버블 정렬

> 시간복잡도 -  O(n^2) 안정적, 제자리

왼쪽(또는 오른쪽)에서부터 모든 인접한 두 값을 비교하여 왼쪽의 값이 더 큰 경우에는 자리를 바꾸는 과정을 반복해서 정렬하는 방식이다.  
자리바꿈이 한 번도 발생하지 않으면 알고리즘을 종료하도록 개선 가능 -> 데이터가 이미 정렬된 경우의 수행 시간은 O(n)

```python
arr = [7,5,9,0,3,1,6,2,4,8]


def bubble_sort(arr):
    for i in range(len(arr)-1,0,-1):
        swapped = False
        for j in range(i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break

    return arr

bubble_sort(arr)
```



# 2. 선택 정렬

> 시간복잡도 - O(n^2) 불안정적, 제자리

주어진 데이터 중에서 **매번 가장 작은 값부터** 차례대로 선택해서 나열하는 방식이다. 데이터의 입력 상태의 민감하지 않는다.

## 선택정렬 1
```python
arr = [7,5,9,0,3,1,6,2,4,8]

def selection_sort(arr):
    for i in range(len(arr)):
        min_index = i
        for j in range(i+1,len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]

    return arr

selection_sort(arr)
```

## 선택정렬 2
```python
arr = [7,5,9,0,3,1,6,2,4,8]

def selection_sort(arr):
    for i in range(len(arr)):
        for j in range(i+1,len(arr)):
            if arr[i]>arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
                
    return arr

selection_sort(arr)
```




# 3. 삽입 정렬

> 시간복잡도 - 최선: O(n)  최악: O(n^2) 안정적, 제자리

주어진 데이터를 하나씩 뽑은 후, **나열된 데이터**들이 항상 정렬된 형태를 가지도록 뽑은 데이터를 바른 위치에 삽입해서 나열하는 방식이다. 특정한 데이터가 적절한 위치에 들어가기 전에, 그 앞까지의 데이터는 **이미 정렬**되어 있다고 가정한다.

```python
arr = [7,5,9,0,3,1,6,2,4,8]

def insertion_sort(arr):
    for i in range(1,len(arr)):
        for j in range(i,0,-1):
            if arr[j] < arr[j-1]:
                arr[j], arr[j-1]=arr[j-1], arr[j]
            else: break

    return arr

insertion_sort(arr)
```

일반적인 경우(위 처럼)에 인덱스 0인 요소를 기준으로 1부터의 요소부터 적절한 위치를 판단한다.




# 4. 셸 정렬

> 시간복잡도 - O(n^2) 불안정적, 제자리

처음에는 멀리 떨어진 두 원소를 비교하여 필요시 위치를 교환하고, **점차적으로** 가까운 위치의 원소를 비교・교환한 뒤, 맨 마지막에는 인적한 원소를 **비교・교환**하는 정렬 방식이다. 삽입 정렬의 단점 보완한다. 간격의 크기를 계산하는 방식에 따라 성능이 다름

```python
arr = [7,5,9,0,3,1,6,2,4,8]
n = 10

def shell_sort(arr, n):

    interval = n // 2  # 간격의 크기 지정
    while interval > 0:
        for i in range(interval, n):  # 간격만큼 떨어진 원소들에 대해 삽입 정렬 수행
            temp = arr[i]
            j = i
            while j >= interval and arr[j - interval] > temp:
                arr[j] = arr[j - interval]
                j -= interval

            arr[j] = temp
        interval //= 2

    return arr

shell_sort(arr, n)
```
