import random
import numpy as np
import sys
import matplotlib.pyplot as plt 

#python Adaline.py datosEntrenamiento.dat datosValidacion.dat datosTest.dat salidasDeseadas.dat "," 10 0.05
#python procesadorDatos.py datos.txt ","

def iniciarHiperparametros(numeroCiclos,raz,atributos,semilla): #inicia los hiperoarametros 
    global razon
    global numCiclos
    global numAtributos
    global pesosIniciales
    razon=raz
    numCiclos=numeroCiclos
    numAtributos=atributos
    pesosIniciales = np.copy(semilla)

def generadorpesosIniciales(cantidadAtributos):  #Genera de forma aleatoria pesos y umbral iniciales
    pesos = []
    
    for i in range(cantidadAtributos): 
        pesos.append(random.uniform(-0.5, 0.5))
    return (np.array(pesos))
    
def leerFicheroDatos(file, separador):    #lee el fichero de datos y devuelve una matriz con todos los datos
    ficheroDatos = open(file, "r")
    data=np.loadtxt(ficheroDatos, dtype=float, delimiter=separador, skiprows=1)
    ficheroDatos.close()
    return(data)

def calculoSalida(patron,pesos):
    copia=pesos.copy()
    #patron[numAtributos-1]=1
    arrayBinario=np.ones(len(patron), dtype=bool)
    arrayBinario[len(arrayBinario)-1]=False
    x=((np.sum(np.multiply(patron, copia, where=arrayBinario, out=copia), dtype=float)))
    ".dot"
  
  
    return(x)

def ajustePesosUmbral(patron,pesos,salidaDeseada,salidaObtenida):
    for i in range(len(pesos)-1): #Actualizamos pesos y umbral ya que el ultimo elemento de patron es 1
        pesos[i] = pesos[i]+(razon*(salidaDeseada-salidaObtenida)*patron[i])
    i+=1
    pesos[i]=pesos[i]+(razon*(salidaDeseada-salidaObtenida))
    

    return (pesos)

def desnormalizacionParametrosSalida(normalizedData,nonNormalizedData): #normaliza datos del array Dim1
	length = len(normalizedData)
	max=-sys.float_info.max
	min=sys.float_info.max
	for i in range (length):
		if(max<nonNormalizedData[i]):
			max=nonNormalizedData[i]
		if(min>nonNormalizedData[i]):
			min=nonNormalizedData[i]
	for i in range (length):        
		normalizedData[i] = min + normalizedData[i]*(max-min)

def imprimirParametros():
    print("Atributos: %i" % numAtributos)
    #print("Numero patrones: %i" % numPatrones)
    print("Numero de ciclos: %i" % numCiclos)
    print("Razon: %f" % razon)

def imprimirGraficasSalidas(x,y1, g1, c1, y2, g2, c2,graphName,xAxis,yAxis,yscale):
	plt.scatter(x, y1,s=1,color=c1,label = g1)
	plt.scatter(x, y2,s=1,color=c2, label = g2)
	plt.yscale(yscale)
	plt.legend() 

	plt.xlabel(xAxis) 
	plt.ylabel(yAxis) 
 
	plt.title(graphName) 
 
	plt.show()

def imprimirGraficasErrores(x, y1, g1, c1, y2, g2, c2, y3, g3, c3, graphName,xAxis,yAxis,yscale):
	plt.plot(x, y1, color=c1,label = g1)
	plt.plot(x, y2, color=c2, label = g2)
	plt.plot(x, y3, color=c3, label = g3)
	plt.yscale(yscale)
	plt.legend() 

	plt.xlabel(xAxis) 
	plt.ylabel(yAxis) 

	plt.title(graphName) 

	plt.show()


#######################---MAIN---########################

if((len(sys.argv) != 8) or (float(sys.argv[7])>(1))): #Comprobamos los parametros introducidos
    print("Parametros incorrectos")
    exit(0)
	
file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
file4 = sys.argv[4]
separador = sys.argv[5]
numeroCiclos=int(sys.argv[6])
razon=float(sys.argv[7])	


entrenamiento=leerFicheroDatos(file1,separador)   #leemos el fichero de entrenamiento
validacion=leerFicheroDatos(file2,separador)   #leemos el fichero de validacion
test=leerFicheroDatos(file3,separador)   #leemos el fichero de test
arraySalidasDeseadas=leerFicheroDatos(file4,separador)   #leemos el fichero de salidas

numeroFilas,numeroAtributos=entrenamiento.shape

pesos=generadorpesosIniciales(numeroAtributos)  #Generamos los pesos iniciales de pesos y umbral

iniciarHiperparametros(numeroCiclos,razon,numeroAtributos,pesos)  #Iniciamos los hiperparametros

imprimirParametros()

#normalizacionParametrosEntrada(data) #Normalizamos los datos (Pasa referencias no copias)
#conjuntos=separacionConjuntosDatos(data,0.6,0.2) #separamos en tres conjuntos los datos

print ("Pesos iniciales:", pesos)

print("-----------------------------------------------------------------------------------------------")
arrayCiclos = []
arrayEntrenamientoMSE = []
arrayValidacionMSE = []
arrayEntrenamientoMAE = []
arrayValidacionMAE = []
#aleatorizar(entrenamiento)

for i in range(numCiclos): #Entrenamiento y test

	sumatorioErrorCuadraticoCicloEntrenamiento=0
	sumatorioErrorAbsolutoCicloEntrenamiento=0
	for j in range(len(entrenamiento)):
		patron=entrenamiento[j]
		'''print "patron"print patron'''
		salidaDeseada=patron[numAtributos-1]
		#print("DESEADAA: %f" %salidaDeseada)
		salidaObtenida=calculoSalida(patron,pesos)
        #print("OBTENIDA: %f" %salidaObtenida)
        #print "pesos antiguo"
        #print pesos
		pesos=ajustePesosUmbral(patron,pesos,salidaDeseada,salidaObtenida)
        #print "pesos nuevos"
        #print pesos
		sumatorioErrorCuadraticoCicloEntrenamiento+=pow((salidaDeseada-salidaObtenida),2)
		sumatorioErrorAbsolutoCicloEntrenamiento+=np.absolute(salidaDeseada-salidaObtenida)    


	
	MSE_Entrenamiento = sumatorioErrorCuadraticoCicloEntrenamiento/len(entrenamiento)
	MAE_Entrenamiento = sumatorioErrorAbsolutoCicloEntrenamiento/len(entrenamiento)
	arrayEntrenamientoMSE.append(MSE_Entrenamiento)
	arrayEntrenamientoMAE.append(MAE_Entrenamiento)
    #print("ENTRENAMIENTO:  MSE: %f/ MAE: %f" % (MSE_Entrenamiento, MAE_Entrenamiento))      

    #INICIO CALCULO ERROR VALIDACION
	sumatorioErrorCuadraticoCicloValidacion=0
	sumatorioErrorAbsolutoCicloValidacion=0
	
	for k in range(len(validacion)):
		patron=validacion[k]
		salidaDeseada=patron[numAtributos-1]
		salidaObtenida=calculoSalida(patron,pesos)
		sumatorioErrorCuadraticoCicloValidacion+=pow((salidaDeseada-salidaObtenida),2)
		sumatorioErrorAbsolutoCicloValidacion+=np.absolute(salidaDeseada-salidaObtenida)
	
	MSE_Validacion=sumatorioErrorCuadraticoCicloValidacion/len(validacion)
	MAE_Validacion=sumatorioErrorAbsolutoCicloValidacion/len(validacion)
	arrayValidacionMSE.append(MSE_Validacion)
	arrayValidacionMAE.append(MAE_Validacion)
	#print("VALIDACION:  MSE: %f/ MAE: %f" % (MSE_Validacion, MAE_Validacion))
    
	arrayCiclos.append(int(i))

#print("DESEADAA: %f" % salidaDeseada)
#print("OBTENIDAA: %f" % salidaObtenida)
#print("ENTRENAMIENTO:  MSE: %f/ MAE: %f" % (MSE_Entrenamiento, MAE_Entrenamiento))
#print("VALIDACION:  MSE: %f/ MAE: %f" % (MSE_Validacion, MAE_Validacion))

#INICIO CALCULO ERROR TEST

sumatorioErrorCuadraticoCicloTest=0
sumatorioErrorAbsolutoCicloTest=0
arraySalidas  = []
arraySalidasObtenidas = []

for l in range(len(test)):  #Test
	patron=test[l]
	salidaDeseada=patron[numAtributos-1]
	salidaObtenida=calculoSalida(patron,pesos)
	sumatorioErrorCuadraticoCicloTest+=pow((salidaDeseada-salidaObtenida),2)
	sumatorioErrorAbsolutoCicloTest+=np.absolute(salidaDeseada-salidaObtenida)
	arraySalidasObtenidas.append(salidaObtenida)
	arraySalidas.append(l)

MSE_Test=sumatorioErrorCuadraticoCicloTest/len(test)
MAE_Test=sumatorioErrorAbsolutoCicloTest/len(test)
arrayTestMSE = np.full(numCiclos,MSE_Test)
arrayTestMAE = np.full(numCiclos,MAE_Test)

print("TEST:  MSE: %f/ MAE: %f" % (MSE_Test, MAE_Test))
print("Modelo final", pesos)

#print("Normalizado>", arraySalidasObtenidas[0])

desnormalizacionParametrosSalida(arraySalidasObtenidas,arraySalidasDeseadas)

#print("NoNormalizado>", arraySalidasObtenidas[0])

salidaAdaline = open("salidaAdaline.txt", 'w')
salidasObtenidasDesnormalizadas = open("salidasObtenidasDesnormalizadas.txt", 'w')

erroresEntrenamientoValidacionMSE = np.stack((arrayEntrenamientoMSE,arrayValidacionMSE), axis=1)
erroresEntrenamientoValidacionMAE = np.stack((arrayEntrenamientoMAE,arrayValidacionMAE), axis=1)

erroresTestMSE = [MSE_Test,MAE_Test]

headerEntrenamientoValidacionMSE = ("Modelo inicial:\t"+ str(pesosIniciales)+"\n\n" 
                                 + "Modelo final\t" + str(pesos)+"\n"
                                 + "\nError MSE (cuadratico medio) de entrenamiento y validacion \n")  
								 
headerEntrenamientoValidacionMAE = "\nError MAE (absoluto) de entrenamiento y validacion \n" 							 

headerTest = "\nError cuadratico medio y absoluto de test"  

np.savetxt(salidaAdaline, erroresEntrenamientoValidacionMSE, delimiter=' / ', newline='\n',  header=headerEntrenamientoValidacionMSE)
np.savetxt(salidaAdaline, erroresEntrenamientoValidacionMAE, delimiter=' / ', newline='\n',  header=headerEntrenamientoValidacionMAE)
np.savetxt(salidaAdaline, erroresTestMSE, delimiter=' / ', newline='\n',header=headerTest)

headerSalidasObtenidasDesnormalizadas = "Salidas obtenidas desnormalizadas"  
np.savetxt(salidasObtenidasDesnormalizadas, arraySalidasObtenidas, newline='\n',header=headerSalidasObtenidasDesnormalizadas)

salidaAdaline.close()
salidasObtenidasDesnormalizadas.close()


imprimirGraficasErrores(arrayCiclos,arrayEntrenamientoMSE,"Entrenamiento", "red",arrayValidacionMSE,"Validacion", "blue", 
                    arrayTestMSE,"Test", "green","MSE Entrenamiento & Validacion & Test", "Ciclos", "Error cuadrático medio","log")

imprimirGraficasSalidas(arraySalidas,arraySalidasDeseadas,"Salida deseada", "red",arraySalidasObtenidas,"Salida Obtenida", "blue","Salidas deseadas y obtenidas", "Ciclos", "Salida","linear")

