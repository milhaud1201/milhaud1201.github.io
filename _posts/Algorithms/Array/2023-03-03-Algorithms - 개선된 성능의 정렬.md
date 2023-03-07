---
title: Algorithms - 개선된 성능의 정렬 (합병/퀵/힙)
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


# 1. 합병 정렬과 퀵 정렬
**분할정복** 방법이 적용된 정렬 알고리즘

## 합병 정렬

> 시간복잡도 - $O(n log n)$ 안정적, 제자리 정렬이 아님

![Alt text](https://upload.wikimedia.org/wikipedia/commons/c/cc/Merge-sort-example-300px.gif)

```python
def merge_sort(arr):
    if len(arr) > 1:

        #  r: 배열이 두 개의 하위 배열로 분할되는 지점
        r = len(arr) // 2
        L = arr[:r]
        M = arr[r:]

        # r로 분할된 두 배열
        merge_sort(L)
        merge_sort(M)

        i = j = k = 0

        # 정렬되지 않은 L과 M 배열을 각각 하나의 원소만 포함한 배열로 분할
        while i < len(L) and j < len(M):
            if L[i] < M[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = M[j]
                j += 1
            k += 1

        # 분할된 배열을 반복해서 병합하고 정렬하며 하나의 정렬된 배열을 만듦
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(M):
            arr[k] = M[j]
            j += 1
            k += 1

    return arr

arr = [7,5,9,0,3,1,6,2,4,8]
merge_sort(arr)
```

## 퀵 정렬

> 시간복잡도 - 평균/최선: $O(n log n)$  최악: $O(n^2)$ 불안정적, 제자리

<iframe width="420" height="315" src="https://www.youtube.com/watch?v=7BDzle2n47c" frameborder="0" allowfullscreen></iframe>

배열의 첫 번째 원소를 **피벗**으로 정한 후 분할 함수 partition()을 통해 왼쪽 부분배열과 오른쪽 부분배열로 **분할**한다. 피벗 앞에는 피벗보다 값이 작은 모든 원소들이 온다. 분할을 마친 뒤 피벗은 더 이상 움직이지 않는다. 분할된 두 개의 작은 리스트에 대해 **재귀**적으로 이 과정을 반복한다.

```python
def partition(arr, start, end):
    pivot = arr[start]
    left = start + 1
    right = end
    done = False
    while not done:
        while left <= right and arr[left] <= pivot:
            left += 1
        while left <= right and pivot <= arr[right]:
            right -= 1
        if right < left:
            done = True
        else:
            arr[left], arr[right] = arr[right], arr[left]
    arr[start], arr[right] = arr[right], arr[start]
    return right


def quick_sort(arr, start, end):
    if start < end:
        pivot = partition(arr, start, end)
        quick_sort(arr, start, pivot - 1)
        quick_sort(arr, pivot + 1, end)
    return arr

arr = [7,5,9,0,3,1,6,2,4,8]
quick_sort(arr, 0, 9)
```

pivot: `7`
[**7**, 5, 4, 0, 3, 1, 6, 2, 9, 8]
pivot: `2`
[**2**, 5, 4, 0, 3, 1, 6, 7, 9, 8]
pivot: `0`
[**0**, 1, 2, 4, 3, 5, 6, 7, 9, 8]
pivot: `4`
[0, 1, 2, **4**, 3, 5, 6, 7, 9, 8]
pivot: `5`
[0, 1, 2, 3, 4, **5**, 6, 7, 9, 8]
pivot: `9`
[0, 1, 2, 3, 4, 5, 6, 7, **9**, 8]

[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



# 2. 힙 정렬

> 시간복잡도 - $O(n log n)$ 안정적, 제자리

각 노드의 값은 자신의 자식 노드의 값보다 크거나 같은 **완전 이진 트리**이다. 일차원 배열로 구현하면 자식 노드 및 부모 노드로의 접근이 용이하다. 초기 힙을 구축한 후, 최댓값 삭제 및 힙으로의 재구성 과정을 반복한다.
* **초기 힙 구축 방법**
    * 입력 배열의 각 원소에 대해 힙에서의 삽입 과정을 반복하는 과정
    * 입력 배열을 우선 완전 이진 트리로 만든 다음 아래에서 위로, 오른쪽에서 왼쪽으로 진행하면서 각 노드에 대해서 힙의 조건을 만족시키는 방법




# 3. 계수 정렬

> 시간복잡도 - $O(n)$ 안정적, 제자리 정렬이 아님

주어진 원소 중에서 자신보다 작거나 같은 값을 갖는 **원소의 개수**를 계산하여 정렬 위치를 찾아 정렬하는 방식이다. 입력 원소의 값이 **어떤 작은 정수 범위 내**에 있다는 것을 알고 있는 경우에만 적용 가능하다.





<br>
<br>
<br>
<br>
<br>
<br>
<br>

** 계속해서 업데이트할 예정입니다 **