# Convolution layer

이 계층에는 CNN만이 사용하는 용어들이 나온다.

각 계층 사이에 3차원 데이터같은 입체적인 데이터가 흐른다는 점이 Affine NN과 다른 점.


## Problem of Affine layer
* Affine(완전 연결) 계층
  * 인접하는 계층의 뉴런이 모두 연결
  * 출력의 수는 임의로 정함.
  * 문제점
    * 데이터의 형상이 무시됨.
      * 이미지면 3차원의 형태로 입력이 들어가야함 (2차원 + 색)
      * 하지만 affine에서는 flatten되어 입력됨.
      * 예: 28 * 28 의 이미지를 1 * 784로 flatten됨.

CNN에서는 Conv layer의 입출력 데이터를 **feature map**이라고 한다.  ${input / output} feature map 으로 부른다.

## Convolution computation
합성곱 연산 -> 이건 익숙하다 전자공학에서 신호처리하면 주구장창하는 거라서 ㅋㅋㅋ (컨볼루션 하지말고 라플라스변환해서 곱하고 역변환하는게 더 편하지만)


아래 행렬을 input data라 하자
~~~
1 2 3 0
0 1 2 3
3 0 1 2
2 3 0 1 
~~~

다음 행렬을 filter라 하자
~~~
2 0 1 
0 1 2 
1 0 2
~~~

이때 두행렬의 컨볼루션 값은 다음과 같다 
~~~
15 16
6  15
~~~

~~~
15 = {1x2 + 2x0 + 3x1} + {0x0 + 1x1 + 2x2} + {3x1 + 0x0 + 1x2}
   = 5                 + 5                 + 5
   
16 = {2x2 + 3x0 + 0x1} + {1x0 + 2x1 + 3x2} + {0x1 + 1x0 + 2x2}
   = 4                 + 8                 + 4
   
6  = {0x2 + 1x0 + 2x1} + {3x0 + 0x1 + 1x2} + {2x1 + 3x0 + 0x2}
   = 2                 + 2                 + 2
   
15 = {1x2 + 2x0 + 3x1} + {0x0 + 1x1 + 2x2} + {3x1 + 0x0 + 1x2}
   = 5                 + 5                 + 5
~~~

1. 필터를 인풋 데이터위에 (좌측 상단 모서리에 맞춰서) 겹치고 각 위치에 있는 값들을 곱하고 모두 더한다.
   * 이 연산과정을 단일 곱셈-누산 (fused multiply-add, FMA)라고 한다.
   * 곱해서 나온 값을 결과로 적는다
2. 이제 필터를 한칸씩 (옆으로/아래로) 민다 그리고 FMA를 진행한다.
   * 곱해서 나온 값은 최초에 계산한 값을 기준으로 shifting한 만큼에 해당되는 위치에 적어준다. 
   * 필터를 아래로 한칸, 우로 한칸 이동한 경우라면 결과도 (1, 1)에 적는다.


affine NN에는 Weight & Bias가 존재했지만 CNN에서는 filter의 params가 weight에 해당되게 된다.

CNN에도 Bias가 존재하는 데, 이 경우는 scalar값이 bias에 적용된다.

이때는 conv 연산으로 나온 행렬의 모든 element에 더해진다.

## Padding
* conv 연산전에 입력데이터 주변을 0 혹은 특정값으로 채우기도 한다. 
* ~~~
  1 2 3 0
  0 1 2 3
  3 0 1 2
  2 3 0 1
  -> (padding: 1)
  0 0 0 0 0 0
  0 1 2 3 0 0
  0 0 1 2 3 0
  0 3 0 1 2 0
  0 2 3 0 1 0
  0 0 0 0 0 0
  ~~~
* 이렇게 채우는 것을 padding이라고 한다
* padding을 적용하는 목적은 주로 출력의 크기를 조정하기 위해서이다.
  * 예를 들어
    * input: (4, 4) 
    * filter: (3, 3)
  * 을 적용하게 되면 
    * output: (2, 2)
  * 로 input보다 크기가 2만큼 줄어든다.
  * conv 연산을 몇번 하냐에 따라 depth가 깊은 NN에서는 문제가 될수 있다. (b/c 어느 시점에는 크기가 1이 됨)
    * 이런 걸 방지하기 위해 패딩을 적용한다. (padding: 1 을 적용하면 (4, 4)로 인풋 크기만큼 그대로 나온다.) 


## Stride
* 이건 쉽다
* 옆으로 **몇** 칸을 밀면서 연산할지 결정.
* 계산식
  * params
    * input: (H, W)
    * filter: (FH, FW)
    * output: (OH, OW)
    * padding: P
    * stride: S
  * OH = (H + 2*P - FH) / S + 1
  * OW = (W + 2*P - FW) / S + 1
  * 주의: 나눗셈 연산이 정수로 나와야 한다.
    * Framework별로 round해주는 등의 방식을 이용하기도한다.

## 3-dim conv
블록으로 생각하면 이해하기 쉽다 (3차원이니...)

* C: Channel
* example 1
  * input: (C, W, H)
  * filter: (C, FH, FW)
  * output: (1, OH, OW)
    * 한장의 feature map이다
    * 다름 말로 channel이 1개인 feature map
* example 2
  * 여러 feature map을 얻기 위해서는 필터를 여러개 적용하면 됨!
  * input: (C, W, H)
  * filter: (FN, C, FH, FW)
  * output: (FN, OH, OW)
* example 3
  * bias는 
    * single filter면 scalar
    * multiple filter면 matrix다.
  * input: (C, W, H)
  * filter: (FN, C, FH, FW)
  * filter_output: (FN, OH, OW)
  * bias: (FN, 1, 1)
  * output: (FN, OH, OW)

## batch processing
batch size를 N이라 하면 위의 예시가 다음과 같아진다.

* input: (N, C, W, H)
* filter: (FN, C, FH, FW)
* filter_output: (N, FN, OH, OW)
* bias: (FN, 1, 1)
* output: (N, FN, OH, OW)

