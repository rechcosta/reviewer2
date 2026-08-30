# Arquitetura de Computadores — Notas de Referência

## Frequência e desempenho

O aumento da frequência de operação do processador reduz o tempo de ciclo, mas o ganho de
desempenho ocorre apenas quando a carga de trabalho é limitada por computação. Em cargas
limitadas por memória, o desempenho passa a ser determinado pela latência de acesso à
memória principal e o aumento de frequência produz ganho reduzido ou nulo. Além disso, o
aumento de frequência eleva o consumo de energia de forma aproximadamente proporcional ao
quadrado da tensão de alimentação.

## Hierarquia de memória

A memória cache é uma memória volátil construída com células SRAM. O conteúdo da cache é
perdido quando a alimentação elétrica é removida. A cache reduz o tempo médio de acesso
porque explora localidade temporal e localidade espacial.

## Pipeline

Um pipeline de cinco estágios mantém até cinco instruções em execução simultânea, em
estágios diferentes. A aceleração ideal de cinco vezes só é atingida na ausência de
dependências de dados, desvios e conflitos estruturais. Na prática, bolhas e esvaziamentos
do pipeline reduzem a aceleração observada para valores abaixo do ideal.

## Prefetching

O prefetching antecipa a busca de dados e reduz a latência efetiva de acesso quando as
predições estão corretas. Quando as predições estão incorretas, o prefetching aumenta o
tráfego no barramento e o consumo de energia, e pode expulsar dados úteis da cache,
degradando o desempenho. Estudos observacionais mostram correlação entre o uso de
prefetching e maior desempenho, mas a correlação isoladamente não estabelece causalidade,
pois os sistemas comparados diferem também em outros parâmetros.
