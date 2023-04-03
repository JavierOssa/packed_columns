import numpy as np
import matplotlib.pyplot as plt
import time
'''
import sys
sys.path.append('C:\\Users\\JUAN FELIPE SARRIA\\Documents\\Académico\\Librerías')
from Determinación_de_raíces import 
'''

' 1. Iteración del punto fijo '

def fixed_point(g, x0, tol, maxIter=100):
    # g: función igualada a x
    # x0: aproximación inicial de la raíz
    # tol: toleracia para el error
    # maxIter: máximo número de iteraciones permitidas
    
    for i in range(1, maxIter):
        #print(x0)
        xi = g(x0)
        if xi != 0:
            ea = abs((xi - x0)/xi)
            if ea < tol:
                break
        x0 = xi
        yi = xi - g(xi)
    return xi, i, yi
    # xi: solución
    # i: número de iteraciones realizadas para lograr la tolerancia
    # yi: verificación de la raíz (será buena si es cercano a cero)

' 2.1. Método de bisección (2i evaluaciones de f1) '

def bisection(f, a, b, tol, maxIter=100):
    # f: función por resolver
    # a: límite inferior del intervalo que contiene la raíz
    # b: límite superior del intervalo que contiene la raíz
    # tol: toleracia para el error
    # maxIter: máximo número de iteraciones permitidas
    
    for i in range(1, maxIter):
        xi = (a+b)/2
        test = np.sign(f(a))*np.sing(f(xi))
        if test < 0:
            b = xi
        elif test > 0:
            a = xi
        else:
            break
        
        if b != 0:
            ea = abs((b - a)/b)
            if ea < tol:
                break
            
    xi = (b + a)/2
    return xi, i, f(xi)
    # xi: solución
    # i: número de iteraciones realizadas para lograr la tolerancia
    # f(xi): verificación de la raíz (será buena si es cercano a cero)
    
' 2.2. Método de bisección optimizado (i+1 evaluaciones de f1) '

def bisection_opt(f, a, b, tol, maxIter=100):
    # f: función por resolver
    # a: límite inferior del intervalo que contiene la raíz
    # b: límite superior del intervalo que contiene la raíz
    # tol: toleracia para el error
    # maxIter: máximo número de iteraciones permitidas
    
    fa = f(a)
    for i in range(1, maxIter):
        xi = (b + a)/2
        fxi = f(xi)
        test = np.sign(fa)*np.sign(fxi)
        if test < 0:
            b = xi
        elif test > 0:
            a = xi
            fa = fxi
        else:
            break
        
        if b != 0:
            ea = abs((b - a)/b)
            if ea < tol:
                break

    xi = (b + a)/2
    return xi, i, f(xi)
    # xi: solución
    # i: número de iteraciones realizadas para lograr la tolerancia
    # f(xi): verificación de la raíz (será buena si es cercano a cero)
    
' 3.1. Método de Newton-Raphson '

def NR_simple(f, df, x0, tol, maxIter=100):
    # f: función por resolver
    # df: derivada de la función por resolver
    # x0: aproximación inicial de la raíz
    # tol: toleracia para el error
    # maxIter: máximo número de iteraciones permitidas
    for i in range(1, maxIter):
        xi = x0 - f(x0)/df(x0) # Error si df1(x0) == 0
        if xi != 0:
            ea = abs((xi - x0)/xi)
            if ea < tol:
                break
        x0 = xi
    return xi, i, f(xi)
    # xi: solución
    # i: número de iteraciones realizadas para lograr la tolerancia
    # f(xi): verificación de la raíz (será buena si es cercano a cero)
    
' 3.2. Método de Newton-Raphson para raíces múltiples '
' Útil cuando la derivada cerca a la raíz es cercana a cero '

def NR_multiple(f, df, d2f, x0, tol, maxIter=100):
    # f: función por resolver
    # df: derivada de la función por resolver
    # d2f: segunda derivada de la funcion por resolver
    # x0: aproximación inicial de la raíz
    # tol: toleracia para el error
    # maxIter: máximo número de iteraciones permitidas
    for i in range(1, maxIter):
        xi = x0 - f(x0)*df(x0)/(df(x0)**2 - f(x0)*d2f(x0))
        if xi != 0:
            ea = abs((xi - x0)/xi)
            if ea < tol:
                break
        x0 = xi
    return xi, i, f(xi)
    # xi: solución
    # i: número de iteraciones realizadas para lograr la tolerancia
    # f(xi): verificación de la raíz (será buena si es cercano a cero)
    
' 4. Método de secante modificado '

def secant(f, x0, h, tol, maxIter=100):
    # f: función por resolver
    # x0: aproximación inicial de la raíz
    # h: diferencial para la derivada numérica
    # tol: toleracia para el error
    # maxIter: máximo número de iteraciones permitidas
    for i in range(1, maxIter):
        df = (-f(x0+2*h) + 8*f(x0+h) - 8*f(x0-h) + f(x0-2*h))/(12*h) # Derivada centrada con precisión O(h^4)
        xi = x0 - f(x0)/df
        if xi != 0:
            ea = abs((xi - x0)/xi)
            if ea < tol:
                break
        x0 = xi
    return xi, i, f(xi)
    # xi: solución
    # i: número de iteraciones realizadas para lograr la tolerancia
    # f(xi): verificación de la raíz (será buena si es cercano a cero)

' 5. Método de Newton-Raphson multivariable '

def NR_multivar(F, DFP, X0, tol, maxIter=100):
    # F: vector (Numpy, nx1) de funciones que reciben un vector (Numpy, nx1) y devuelven un escalar
    # DFP: función que recibe un vector (Numpy, nx1) y devuelve la matriz (Numpy, nxn) de derivadas parciales evaluada en el vector
    # X0: vector de aproximación inicial de la raíz (Numpy, nx1)
    # tol: toleracia para el error
    # maxIter: máximo número de iteraciones permitidas

    n = X0.size # Número de variables/ecuaciones
    Fi = np.zeros((n, 1))
    J = np.zeros((n, n))

    for i in range(1, maxIter):
        for j in range(0, n):
            Fi[j, 0] = F[j, 0](X0)
        J = DFP(X0)
        H = np.linalg.solve(J, -Fi)
        Xi = X0 + H

        if abs(np.max(H/Xi)) < tol:
            break
        
        X0 = Xi.copy()

    return Xi, i, Fi
    # Xi: vector de solución al sistema de ecuaciones (Numpy, nx1)
    # i: número de iteraciones necesarias para lograr la tolerancia
    # Fi: imagen del vector solución (Numpy, nx1). La solución es adecuada si es cercano a cero.
    
''' Template for general Newton-Raphson method
X0 = np.array([])

def f1(X):
    x, y = X[0,0], X[1, 0]
    return 

def f2(X):
    x, y = X[0,0], X[1, 0]
    return 
    
Fs = np.array([[f1], [f2]])

def dfp(X):
    x, y = X[0,0], X[1, 0]
    return np.array([])

print(Newton_general(X0, Fs, dfp, 0.5e-5))
'''

def secant_multivar(F, X0, h, tol, maxIter=100):
    # F: vector (Numpy) de funciones que reciben un vector (Numpy) y devuelven un escalar
    # X0: vector de aproximación inicial de la raíz (Numpy)
    # h: diferencial para las derivadas parciales
    # tol: toleracia para el error
    # maxIter: máximo número de iteraciones permitidas

    def centered_dfp(f, var, X):
        # f: función a derivar parcialmente
        # var: número de variable sobre la cual derivar
        # X: vector a evaluar
        H = np.zeros((X.size, 1))
        H[var, 0] = h
        return (f(X + H) - f(X - H))/(2*h)
        # Devuelve la derivada parcial de f respecto a var evaluada en X

    n = X0.size # Número de variables/ecuaciones
    F = F.reshape((n, 1))
    X0 = X0.reshape((n, 1))
    Fi = np.zeros((n, 1))
    J = np.zeros((n, n))

    for i in range(1, maxIter):
        for j in range(0, n):
            Fi[j, 0] = F[j, 0](X0)
        for k in range(0, n):
            for j in range(0, n):
                J[k, j] = centered_dfp(F[k, 0], j, X0)
        Hi = np.linalg.solve(J, -Fi)
        Xi = X0 + Hi

        if abs(np.max(Hi/Xi)) < tol:
            break
        
        X0 = Xi.copy()

    return Xi.reshape(n), i, Fi.reshape(n)
    # Xi: vector de solución al sistema de ecuaciones (Numpy)
    # i: número de iteraciones necesarias para lograr la tolerancia
    # Fi: imagen del vector solución (Numpy). La solución es adecuada si es cercano a cero.

''' Template for multivariable secant method

def f1(X):
    x, y = X
    return x**2 - np.log(y)

def f2(X):
    x, y = X
    return x**3 - 5*y + 10

F = np.array([f1, f2])
X0 = np.array([1.5, 5])

'''

' 6. Híbrido de secante - bisección '

def secant_bisection(f, a, b, h, tol, maxIter=100):
    # f: función por resolver
    # a: límite inferior del intervalo que contiene la raíz
    # b: límite superior del intervalo que contiene la raíz
    # h: paso para la derivada numérica
    # tol: toleracia para el error
    # maxIter: máximo número de iteraciones permitidas

    x0 = a
    
    for i in range(1, maxIter):
        fi = f(x0)
        dfi = (f(x0 + h) - f(x0 - h))/(2*h)
        test1 = dfi > 0 and dfi*(x0-a) > fi and fi > dfi*(x0-b)
        test2 = dfi < 0 and dfi*(x0-a) < fi and fi < dfi*(x0-b)
        if test1 or test2: # xi cae en [a, b] entonces se usa secante
            xi = x0 - fi/dfi
        else: # xi no cae en [a, b] entonces se usa bisección
            xi = (a+b)/2
            test3 = np.sign(f(xi))*np.sign(f(a))
            if test3 < 0:
                b = xi
            elif test3 > 0:
                a = xi

        ea = abs((xi-x0)/xi)
        if ea < tol:
            break
        
        x0 = xi
    return xi, i, fi

# Cardano's method for finding roots of third-degree polynomials

def Cardano(a0, a1, a2):
    Q = (a2**2 - 3*a1)/9
    R = (2*a2**3 - 9*a2*a1 + 27*a0)/27
    M = R**2 - 4*Q**3 # Discriminante
    if M < 0: # Hay tres raíces reales
        theta = np.arccos(-R/(2*Q**(3/2)))/3
        z1 = 2*np.sqrt(Q)*np.cos(theta) - a2/3
        z2 = 2*np.sqrt(Q)*np.cos(theta + 2*np.pi/3) - a2/3
        z3 = 2*np.sqrt(Q)*np.cos(theta + 4*np.pi/3) - a2/3
        z = np.array([z2, z3, z1]) # En orden ascendente
    else: # Hay una raíz real
        S = np.cbrt((-R + np.sqrt(M))/2)
        T = np.cbrt((-R - np.sqrt(M))/2)
        z = S + T - a2/3
    return z
