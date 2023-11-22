#Alle Imports
import numpy as np
from matplotlib import pyplot as plt
import array 
import math as math
import random

#Numpy
array1 = np.array([1,4,0,1,0,2,1,3]).reshape(2,4)
#[[1 4 0 1]
# [0 2 1 3]]
array2 = np.array([1,0,3,0,1,1,3,4,1,2,0,1]).reshape(4,3)
#[[1 0 3]
# [0 1 1]
# [3 4 1]
# [2 0 1]]

my_np_array = np.zeros(shape=(10), dtype=int)
#[0 0 0 0 0 0 0 0 0 0]

my_np_array = np.ones(shape=(3, 5), dtype=float)
#[[1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1.]]
my_np_array = np.full(shape=(3, 5), fill_value=3.14)
#[[3.14 3.14 3.14 3.14 3.14]
# [3.14 3.14 3.14 3.14 3.14]
# [3.14 3.14 3.14 3.14 3.14]]

my_np_array = np.arange(start=0, stop=20, step=2)
#[ 0  2  4  6  8 10 12 14 16 18]

my_np_array = np.linspace(start=0, stop=1, num=5)
#[0.   0.25 0.5  0.75 1.  ]

my_np_array = np.eye(N=3)
#[[1. 0. 0.]
# [0. 1. 0.]
# [0. 0. 1.]]

my_np_array = np.diag([1, 2, 3])
#[[1 0 0]
# [0 2 0]
# [0 0 3]]

my_np_array = np.random.randint(low=0, high=10, size=(3, 3))
#[[4 7 8]
# [2 9 9]
# [4 4 7]]

my_np_array = np.random.random(size=(3, 3))
#[[0.41085025 0.76310947 0.31047541]
# [0.5874872  0.23099643 0.68777054]
# [0.13104996 0.22085346 0.76468711]]

#-----------------Ufunctions-------------------------------------


np.matmul(my_np_array,array2)   #Multiplizieren von Matrizen
np.multiply(my_np_array,array2) #Skalarprodukt von Matrizen
np.add(my_np_array,array2)      #Matrizen addieren

#| add        | Adds, element-wise                                            |
#| subtract   | Subtracts, element-wise                                       |
#| multiply   | Multiplies, element-wise                                      |
#| matmul     | Matrix product of two arrays                                  |
#| divide     | Returns a true division of the inputs, element-wise           |
#| negative   | Numerical negative, element-wise                              |
#| positive   | Numerical positive, element-wise                              |
#| mod        | Return, element-wise remainder of division                    |
#| absolute   | Calculate the absolute value, element-wise                    |
#| fabs       | Compute the absolute values, element-wise                     |
#| sign       | Returns an, element-wise indication of the sign of a number   |
#| exp        | Calculate the exponential of all elements in the input array  |
#| log        | Natural logarithm, element-wise                               |
#| sqrt       | Return the non-negative square-root of an array, element-wise |
#| square     | Return the, element-wise square of the input                  |
#| reciprocal | Return the reciprocal of the argument, element-wise           |
#| gcd        | Returns the greatest common divisor of \|x1\| and \|x2\|      |
#| lcm        | Returns the lowest common multiple of \|x1\| and \|x2\|       |


#-----------------------Trigonometrische functions-------------------------------#
#| sin  | Trigonometric sine, element-wise |
#| cos  | Cosine, element-wise             |
#| tan  | Compute tangent, element-wise    |

#------------------------Vergleichsfuntkionen------------------------------------#

#| greater       | Return the truth value of (x1 > x2), element-wise  |
#| greater_equal | Return the truth value of (x1 >= x2), element-wise |
#| less          | Return the truth value of (x1 < x2), element-wise  |
#| less_equal    | Return the truth value of (x1 <= x2), element-wise |
#| not_equal     | Return (x1 != x2), element-wise                    |
#| equal         | Return (x1 == x2), element-wise                    |

#-------------------Slice zugriffe --------------------------#

M[::2,::2]                                      #jedes zweite Element
M[von:bis:schrittweite,von:bis:schrittweite]    #Links sind Zeilen und rechts sind spalten

i=np.triu_indices(9,1)                            #von rechts oben beginnend bis zum zweiten wert 0 ist die hälfte
i=np.tril_indices(9,-1)                           #von links unten beginnend bis zum zweiten wert 0 ist die hälfte

M[i]=0

print(np.count_nonzero(M<0))                    #Zählt wie viele Zahlen kleiner als 0 enthalten sind

print(np.count_nonzero(N<0))                    #Zählt wie viele Zahlen größer als 0 enthalten sind

M[M<0]=0                                        #setzt alles was kleiner als 0 ist mit 0

N[N<0]=0                                        #setzt alles was größer als 0 ist mit 0 

M = np.multiply(M,np.pi)                        #Matrix mit pi multiplizieren

M = np.cos(M)                                   #Cosinusfunktion auf alle elemente des arrays anwenden

det = np.linalg.det(M)                          #Berechnung der Determinante (Übergabewert ist Matritze)


#----------------------------------Matplotlib-------------------------------#

x=[n for n in range(-5,5,1)]
y=[2*n +3 for n in x]                         #Geradengleichung

plt.plot(x,y)                                 #Werte für die Koordinaten x-Achse und y-Achse
plt.show 


#Beschriftung der Graphen
x=[n for n in range(-5,5,1)]
y=[-2*n -3 for n in x]                         

plt.plot(x,y)
plt.xlabel("X-Werte")                       #Beschriftung X-Achse
plt.ylabel("Y-Werte")                       #Beschriftung y-Achse
plt.title("Funktion f(x)=-2x-3")            #Beschriftung Titel
plt.grid()                                  #Anzeige Hilfslinien
plt.show                                    #Zeigt Graphen an 

plt.plot(x,y3, color='black')               #Linienfarbe veränder in Scharz
plt.plot(x,y2, color='red')                 # oder rot

plt.grid(color='blue', linestyle='dashed', linewidth=0.5)   #Gird modifizieren linienfarbe blau gepuntet und 0.5 dicke


#Subplot
x=[n for n in range(-5,5,1)]
y=[2*n +3 for n in x]                         

plt.subplot(1,2,1)              # plt.subplot(anzahl zeilen, anzahl spalten, nummer des bildes)
plt.plot(x,y)                                 

x=[n for n in range(-5,5,1)]
y=[-2*n -3 for n in x]                         
plt.subplot(1,2,2)
plt.plot(x,y)
plt.xlabel("X-Werte")                       
plt.ylabel("Y-Werte")                       
plt.title("Funktion f(x)=-2x-3")            
plt.grid()                                  
plt.show()



#Zeigt Scatter an sind Punkt Graphen
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y)
#plt.grid()
plt.show()


#Zeigt Balkendiagramm an 
x = np.random.normal(170, 10, 250)              #random.normal liefert uns eine Normalverteilte 
                                                #Zufallsvariable, dazu kommen wir in der nächsten 
                                                #Stunde
plt.hist(x)
plt.show() 


x = [n/10 for n in range(-100,101,1)]           #Erstellen von zahlen -10-10 Schrittweite 0,1
y = [n**2 + 3*n +3 for n in x]                  #Alle Werte von x werden in Gleichung x² + 3x eingesetzt


#Expotential Funktion

x = [n/10 for n in range(-100,101,1)]
y = [ math.exp(n) for n in x]

plt.plot(x,y)
plt.show()

#Logarithmus Funktion
x = [n/10 for n in range(-100,101,1)]
y = [ math.log2(n+11) for n in x]

plt.plot(x,y, 'red')
plt.show()


zufall4000 = np.random.randint(0,25,4000)   

unique, counts = np.unique(zufall4000,return_counts=True)
result = np.column_stack((unique, counts))
formatstring = "{: >4} | {: ^4}"
for paare in result:
    print(formatstring.format(paare[0],paare[1]))

#Ende