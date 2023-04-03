import numpy as np
'''
import sys
sys.path.append('C:\\Users\\JUAN FELIPE SARRIA\\Documents\\Académico\\Librerías')
'''

' Tomando el extremo izquierdo (n evaluaciones de f)'

def Int_I(f, a, b, n):
    # f: función a integrar numéricamente
    # a, b: intervalo de integración
    # n: número de particiones
    h = (b - a)/n
    sum = 0
    for i in range(0, n):
        sum += f(a + i*h)
    ans = h*sum
    return ans

' Tomando el extremo derecho (n evaluaciones de f)'

def Int_D(f, a, b, n):
    # f: función a integrar numéricamente
    # a, b: intervalo de integración
    # n: número de particiones
    h = (b - a)/n
    sum = 0
    for i in range(1, n+1):
        sum += f(a + i*h)
    ans = h*sum
    return ans

' Tomando el centro (n evaluaciones de f)'

def Int_C(f, a, b, n):
    # f: función a integrar numéricamente
    # a, b: intervalo de integración
    # n: número de particiones
    h = (b - a)/n
    sum = 0
    a2 = a + h/2
    for i in range(0, n):
        sum += f(a2 + i*h)
    ans = h*sum
    return ans

' Tomando el centro, sumando en sentido inverso '
# Se utiliza cuando la función es decreciente y f(b) es cercano a 0
# Al sumar los números más pequeños primero, se disminuye el error de redondeo

def Int_C_Inv(f, a, b, n):
    # f: función a integrar numéricamente
    # a, b: intervalo de integración
    # n: número de particiones
    h = (b - a)/n
    sum = 0
    a2 = a + h/2
    for i in range(n - 1, -1, -1):
        sum += h*f(a2 + i*h)
    return sum

' Regla del trapecio (n + 1 evaluaciones de f)'

def Int_Trap(f, a, b, n):
    # f: función a integrar numéricamente
    # a, b: intervalo de integración
    # n: número de particiones
    h = (b - a)/n
    sum = 0
    for i in range(1, n):
        sum += f(a + i*h)
    ans = h*(f(a) + 2*sum + f(b))/2
    return ans

' Regla de Simpson 1/3 (n + 1 evaluaciones de f) '

def Int_Simpson13(f, a, b, n):
    # f: función a integrar numéricamente
    # a, b: intérvalo de integración
    # n: número de particiones (debe ser par)

    h = (b - a)/n

    sum1 = 0
    sum2 = 0

    for i in range(1, n):
        if i%2 == 1:
            sum1 += f(a + h*i)
        else:
            sum2 += f(a + h*i)

    ans = (f(a) + 4*sum1 + 2*sum2 + f(b))*h/3

    return ans

' Regla de Simpson 3/8 (n + 1 evaluaciones de f) '

def Int_Simpson38(f, a, b, n):
    # f: función a integrar numéricamente
    # a, b: intérvalo de integración
    # n: número de particiones (debe ser múltiplo de 3)

    h = (b - a)/n

    sum1 = 0
    sum2 = 0

    for i in range(1, n):
        if i%3 == 0:
            sum1 += f(a + h*i)
        else:
            sum2 += f(a + h*i)

    ans = (f(a) + 2*sum1 + 3*sum2 + f(b))*3/8*h

    return ans

' Rectángulos por la izquierda y por la derecha con acotación del error estimado '

def Int_ID_Acot(f, a, b, tol):
    
    # Integral por la izquierda
    
    def Int_I(f, a, b, n):
        h = (b - a)/n
        sum = 0
        for i in range(0, n):
            sum += f(a + i*h)
        ans = h*sum
        return ans
    
    # Integral por la derecha

    def Int_D(f, a, b, n):
        h = (b - a)/n
        sum = 0
        for i in range(1, n+1):
            sum += f(a + i*h)
        ans = h*sum
        return ans

    # Estimación inicial del error con n=100
    
    AI = Int_I(f, a, b, 100)
    AD = Int_D(f, a, b, 100)
    Ea = abs(AI - AD)/2
    h = (b - a)/100

    # Cálculo de la integral con el error deseado

    h = h*tol/Ea
    n = int((b - a)/h) + 1
    AI = Int_I(f, a, b, n)
    AD = Int_D(f, a, b, n)

    A = (AI + AD)/2
    
    return A, n
    # A: valor de la integral
    # n: número de particiones necesarias para lograr la tolerancia

' Regla del trapecio con acotación del error '

def Int_Trap_Acot(f, a, b, tol):

    # Segunda derivada
    
    def df2_C(f, x, h):
        df2 = (f(x + h) - 2*f(x) + f(x - h))/(h**2) # Centrada O(h^2)    
        return df2
    
    # Regla del trapecio

    def Int_Trap(f, a, b, n):
            h = (b - a)/n
            sum = 0
            for i in range(1, n):
                sum += f(a + i*h)
            ans = h*(f(a) + 2*sum + f(b))/2
            return ans  
    
    # Estimación de máxima segunda derivada

    df2List = np.zeros(0)

    h = (b - a)/100
    for i in range(0, 100):
        df2i = df2_C(f, a + h*i, 0.01)
        df2List = np.append(df2List, df2i)

        if i == 0:
            df2Max = df2i
        else:
            if abs(df2i) > abs(df2Max):
                df2Max = df2i

    # Cálculo de la integral con el nuevo h

    h = (abs(12*tol/((b - a)*df2Max)))**(1/2)
    n = int((b - a)/h) + 1

    A = Int_Trap(f, a, b, n)

    return A, n
    # A: valor de la integral
    # n: número de particiones necesarias para lograr la tolerancia

' Cuadratura de Gauss-Legendre (n evaluaciones de f)'

def Gauss_Legendre(f, a, b, n, roots=False):
    # f: función a integrar numéricamente
    # a, b: límites inferior y superior del intervalo de integración, respectivamente
    # n: número de evaluaciones
    
    def eval(grad):
        if not(roots):
            xList, wList = np.polynomial.legendre.leggauss(grad) # Si se va a integrar varias veces, mejor sacarlo de la función
        else:
            xList, wList = roots
        a1 = (b - a)/2
        a0 = (a + b)/2
        sum = 0
        for i in range(0, grad):
            sum += wList[i]*f(a1*xList[i] + a0)
        ans = a1*sum
        return ans

    Int = eval(n)
    erAprox = abs((Int - eval(n - 1))/Int) # Estimación según la diferencia respecto a la integral con n - 1 evaluaciones

    return Int, erAprox
    # Int: valor de la integral
    # erAprox: error relativo aproximado
