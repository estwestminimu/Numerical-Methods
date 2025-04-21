#Szymon Tomaszewski
import numpy as numpy
import matplotlib.pyplot as plt



#64 BIT
x_64=0.2
h_64 = numpy.logspace(-20, 0, 400, dtype=numpy.float64)
# wyliczone recznie
derivativeF_64 = numpy.float64(3 * numpy.cos(x_64 ** 3) * (x_64 ** 2))


#32 BIT
x_32 = numpy.float32(0.2)
h_32 = numpy.logspace(-10, 0, 400, dtype=numpy.float32)
# wyliczone recznie
derivativeF_32 = numpy.float32(3 * numpy.cos(x_32 ** numpy.float32(3)) * (x_32 ** numpy.float32(2)))





#FUNKCJE
def defaultF(x):
    return numpy.sin(x**3)

def symmetricDerivative(x,h):
    return (defaultF(x+h)-defaultF(x))/h


def DividedDifferences(x, h):
    return (defaultF(x+h) - defaultF(x - h)) / (2 * h)

#w przod
def analyze_symmetricDerivative( x, hs, der):
    errors= []

    for h in hs:
        approx = symmetricDerivative( x, h)
        error = numpy.abs(approx - der)
        errors.append(error)
    return errors;

#centralna
def analyze_DividedDifferences( x, hs, der):
    errors= []

    for h in hs:
        approx = DividedDifferences( x, h)
        error = numpy.abs(approx - der)
        errors.append(error)
    return errors;





errors_symmetric_32 = analyze_symmetricDerivative( x_32, h_32, derivativeF_32)
errors_divided_32 = analyze_DividedDifferences( x_32, h_32, derivativeF_32)
errors_symmetric_64 = analyze_symmetricDerivative( x_64, h_64, derivativeF_64)
errors_divided_64 = analyze_DividedDifferences( x_64, h_64, derivativeF_64)

# 32Bit  w przod
plt.figure(figsize=(8, 6))
plt.loglog(h_32, errors_symmetric_32, linestyle='--', color='b')
plt.xlabel('h')
plt.ylabel('E(h)')
plt.title('Błąd (f(x+h)-f(x))/h 32bit')
plt.grid(True)
plt.show()


# 64 bit w przod
plt.figure(figsize=(8, 6))
plt.loglog(h_64, errors_symmetric_64, linestyle='--', color='b')
plt.xlabel('h')
plt.ylabel('E(h)')
plt.title('Błąd (f(x+h)-f(x))/h 64bit')
plt.grid(True)
plt.show()


# 32 bit centralna
plt.figure(figsize=(8, 6))
plt.loglog(h_32, errors_divided_32, linestyle='--', color='b')
plt.xlabel('h')
plt.ylabel('E(h)')
plt.title('Błąd (f(x+h)-f(x-h))/2h 32bit')
plt.grid(True)
plt.show()

# 64 bit Roznica centralna
plt.figure(figsize=(8, 6))
plt.loglog(h_64, errors_divided_64, linestyle='--', color='b')
plt.xlabel('h')
plt.ylabel('E(h)')
plt.title('Błąd (f(x+h)-f(x-h))/2h 64bit')
plt.grid(True)
plt.show()