import numpy as np
import sys

def leerFicheroDatos(file, separador):    #lee el fichero de datos y devuelve una matriz con todos los datos
    ficheroDatos = open(file, "r")
    data=np.loadtxt(ficheroDatos, dtype=float, delimiter=separador, skiprows=1)
    ficheroDatos.close()
    return(data)

def iniciarHiperparametros(numeroFilas,atributos): #inicia los hiperoarametros 

    global numPatrones
    global numAtributos

    numPatrones=numeroFilas
    numAtributos=atributos

def normalizacionParametrosEntrada(data): #normaliza la matriz de datos
    
    for j in range (numAtributos):
        max=-sys.float_info.max
        min=sys.float_info.max
        for i in range (numPatrones):
            if(max<data[i][j]):
                max=data[i][j]
            if(min>data[i][j]):
                min=data[i][j]
        for i in range (numPatrones):        
            data[i][j]=(data[i][j]-min)/(max-min)
       
def separacionConjuntosDatos(data,porcentajeEntrenamiento,porcentajeValidacion): #Divide los datos en tres conjuntos segun los porcentajes recividos
    patronesEntrenamiento=int(porcentajeEntrenamiento*numPatrones)
    patronesValidacion=int(porcentajeValidacion*numPatrones)
    

    datosEntrenamiento = data[0:(patronesEntrenamiento),0:numAtributos]
    datosValidacion = data[patronesEntrenamiento:patronesValidacion+patronesEntrenamiento,0:numAtributos]
    datosTest = data[patronesValidacion+patronesEntrenamiento:numPatrones,0:numAtributos]
    '''  
    numeroFilas=np.shape(datosEntrenamiento)[0]   
    print(numeroFilas)
    numeroFilas=np.shape(datosValidacion)[0] 
    print(numeroFilas)
    numeroFilas=np.shape(datosTest)[0] 
    print(numeroFilas)   
    print(patronesEntrenamiento)
    print(patronesValidacion)
    print(datosEntrenamiento[patronesEntrenamiento-1])
    print(datosValidacion[0])
    print(datosValidacion[patronesValidacion-1])
    print(datosTest[0])
    print(datosTest[3400])
    '''
    conjuntos=[datosEntrenamiento,datosValidacion,datosTest]
    return(conjuntos)

def aleatorizar(data): #recibe un conjunto de datos y desordena las filas
    np.random.shuffle(data)

#--------Main---------

if((len(sys.argv) != 3)): #Comprobamos los parametros introducidos
    print("Parametros incorrectos")
    exit(0)
	
file = sys.argv[1]
separador = sys.argv[2]

datos = leerFicheroDatos(file,separador)

numeroFilas,numeroAtributos=datos.shape

iniciarHiperparametros(numeroFilas,numeroAtributos)

conjuntos=separacionConjuntosDatos(datos,0.6,0.2) #separamos en tres conjuntos los datos

salidasDeseadas = conjuntos[2][:,numeroAtributos-1].copy()

normalizacionParametrosEntrada(datos)
aleatorizar(datos)

entrenamiento = open("datosEntrenamiento.dat", 'w')
validacion = open("datosValidacion.dat", 'w')
test = open("datosTest.dat", 'w')
salidas = open("salidasDeseadas.dat", 'w')

np.savetxt(entrenamiento, conjuntos[0], delimiter=',', newline='\n',  header="Datos entrenamiento normalizados y aleatorizados")
np.savetxt(validacion, conjuntos[1], delimiter=',', newline='\n',  header="Datos validacion normalizados y aleatorizados")
np.savetxt(test, conjuntos[2], delimiter=',', newline='\n',  header="Datos entrenamiento normalizados y aleatorizados")
np.savetxt(salidas, salidasDeseadas, delimiter=',', newline='\n',  header="Salidas")

entrenamiento.close()
validacion.close()
test.close()