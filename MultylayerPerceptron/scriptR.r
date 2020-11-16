library(RSNNS)

## funcion que calcula el error cuadratico medio
MSE <- function(pred,obs) {sum((pred-obs)^2)/length(obs)}
MAE <- function(pred,obs) {sum(abs(pred-obs)/length(obs))}

#CARGA DE DATOS

# IMPORTANTE -> para ejecutar con R eliminar "introduccion"
# se supone que los ficheros tienen encabezados

trainSet <- read.csv("datosEntrenamiento.dat",dec=".",sep=",",header = F)
validSet <- read.csv( "datosValidacion.dat",dec=".",sep=",",header = F)
testSet  <- read.csv("datosTest.dat",dec=".",sep=",",header = F)

#trainSet <- read.table("trainParab.dat")
#validSet <- read.table( "testParab.dat")
#testSet <- read.table( "testParab.dat")

salida <- ncol (trainSet)   #num de la columna de salida

#SELECCION DE LOS PARAMETROS
#Topologia: 1 o 2 capas ocultas. Numero de neuronas testeo

topologia        <- c(8,8) #PARAMETRO DEL TIPO c(A,B,C,...,X) A SIENDO LAS NEURONAS EN LA CAPA OCULTA 1, B LA CAPA 2 ...
razonAprendizaje <- 0.5 #NUMERO REAL ENTRE 0 y 1
ciclosMaximos    <- 10 #NUMERO ENTERO MAYOR QUE 0

#EJECUCION DEL APRENDIZAJE Y GENERACION DEL MODELO

set.seed(1)
model <- mlp(x= trainSet[,-salida],
             y= trainSet[, salida],
             inputsTest=  validSet[,-salida],
             targetsTest= validSet[, salida],
             size= topologia,
             maxit=ciclosMaximos,
             learnFuncParams=c(razonAprendizaje),
             shufflePatterns = F
             )

#GRAFICO DE LA EVOLUCION DEL ERROR
plotIterativeError(model)

# DATAFRAME CON LOS ERRORES POR CICLo: de entrenamiento y de validacion
iterativeErrors <- data.frame(MSETrain= (model$IterativeFitError/ nrow(trainSet)),
                              MSEValid= (model$IterativeTestError/nrow(validSet)))

######################################################################
#SE OBTIENE EL NuMERO DE CICLOS DONDE EL ERROR DE VALIDACION ES MINIMO
#######################################################################

nuevosCiclos <- which.min(model$IterativeTestError)

#ENTRENAMOS LA MISMA RED CON LAS ITERACIONES QUE GENERAN MENOR ERROR DE VALIDACION
set.seed(1)
model <- mlp(x= trainSet[,-salida],
             y= trainSet[, salida],
             inputsTest=  validSet[,-salida],
             targetsTest= validSet[, salida],
             size= topologia,
             maxit=nuevosCiclos,
             learnFuncParams=c(razonAprendizaje),
             shufflePatterns = F
)
#GRAFICO DE LA EVOLUCION DEL ERROR
plotIterativeError(model)

iterativeErrors1 <- data.frame(MSETrain= (model$IterativeFitError/ nrow(trainSet)),
                              MSEValid= (model$IterativeTestError/nrow(validSet)))

#CALCULO DE PREDICCIONES
prediccionesTrain <- predict(model,trainSet[,-salida])
prediccionesValid <- predict(model,validSet[,-salida])
prediccionesTest  <- predict(model, testSet[,-salida])

#CALCULO DE LOS ERRORES
errorsMSE <- c(TrainMSE= MSE(pred= prediccionesTrain,obs= trainSet[,salida]),
            ValidMSE= MSE(pred= prediccionesValid,obs= validSet[,salida]),
            TestMSE=  MSE(pred= prediccionesTest ,obs=  testSet[,salida]))

errorsMAE <- c(TrainMAE= MAE(pred= prediccionesTrain,obs= trainSet[,salida]),
            ValidMAE= MAE(pred= prediccionesValid,obs= validSet[,salida]),
            TestMAE=  MAE(pred= prediccionesTest ,obs=  testSet[,salida]))
errorsMSE

errorsMAE

#SALIDAS DE LA RED
outputsTrain <- data.frame(pred= prediccionesTrain,obs= trainSet[,salida])
outputsValid <- data.frame(pred= prediccionesValid,obs= validSet[,salida])
outputsTest  <- data.frame(pred= prediccionesTest, obs=  testSet[,salida])


#GUARDANDO RESULTADOS
saveRDS(model,"nnet.rds")
#write.csv2(errorsMSE,"finalErrorsMSE.csv")
#write.csv2(errorsMAE,"finalErrorsMAE.csv")
#write.csv2(iterativeErrors,"iterativeErrors.csv")
#write.csv2(outputsTrain,"netOutputsTrain.csv")
#write.csv2(outputsValid,"netOutputsValid.csv")
#write.csv2(outputsTest, "netOutputsTest.csv")

# #############
# colnames(trainSet)=c("x","y","z")
# head(trainSet)
# modelo=lm(z~x+y, trainSet)
# 
# summary(modelo)
# mselin <- mean(modelo$residuals^2)
# mselin   #error mse
# 
# Fuente: https://www.i-ciencias.com/pregunta/89240/como-obtener-el-valor-del-error-cuadratico-medio-de-una-regresion-lineal-en-r
